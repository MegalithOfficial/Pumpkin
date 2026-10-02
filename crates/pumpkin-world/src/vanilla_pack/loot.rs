//! Vanilla loot tables, converted from the jar's json while the pack is prepared.
//!
//! Predicate, item tag and nested loot table references are resolved during conversion, so a
//! table in the pack is self contained and decoding it needs nothing else.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use dashmap::DashMap;
use pumpkin_util::loot_table::{
    DynamicLootCondition as Condition, DynamicLootEntry, DynamicLootPool, DynamicLootTable,
    LootBonusFormula,
};
use serde::Deserialize;

use super::{PackError, ResourceKind, installed};

// guards against reference cycles between nested loot tables
const MAX_NESTING: usize = 5;

/// Jar json that loot tables refer to, keyed by path under `data/minecraft/<dir>/` without
/// the extension, like `blocks/stone`.
#[derive(Default)]
pub struct LootSources {
    pub loot_tables: HashMap<String, String>,
    pub predicates: HashMap<String, String>,
    pub item_tags: HashMap<String, String>,
}

impl LootSources {
    /// Converts every collected loot table into `(path, table)` pairs, or names the one that
    /// failed.
    pub fn convert_all(
        &self,
    ) -> Result<Vec<(String, DynamicLootTable)>, (String, serde_json::Error)> {
        self.loot_tables
            .iter()
            .map(|(path, json)| match self.convert(json) {
                Ok(table) => Ok((path.clone(), table)),
                Err(e) => Err((path.clone(), e)),
            })
            .collect()
    }

    fn convert(&self, json: &str) -> Result<DynamicLootTable, serde_json::Error> {
        let table: TableJson = serde_json::from_str(json)?;
        let pools = table
            .pools
            .iter()
            .map(|pool| {
                let mut entries = Vec::new();
                let mut empty_weight = 0;
                for entry in &pool.entries {
                    self.extract_entries(
                        entry,
                        Condition::None,
                        &mut entries,
                        &mut empty_weight,
                        0,
                    );
                }
                DynamicLootPool {
                    entries,
                    min_rolls: pool.rolls.min(),
                    max_rolls: pool.rolls.max(),
                    empty_weight,
                    condition: self.condition_of(pool.condition.as_ref()),
                }
            })
            .collect();
        Ok(DynamicLootTable { pools })
    }

    fn lookup<'a>(map: &'a HashMap<String, String>, id: &str) -> Option<&'a str> {
        map.get(id.strip_prefix("minecraft:").unwrap_or(id))
            .map(String::as_str)
    }

    fn condition_of(&self, value: Option<&ConditionValue>) -> Condition {
        value.map_or(Condition::None, |v| self.resolve_condition(v))
    }

    fn resolve_condition(&self, value: &ConditionValue) -> Condition {
        match value {
            ConditionValue::Inline(cond) => self.parse_condition(cond),
            ConditionValue::Reference(id) => Self::lookup(&self.predicates, id)
                .and_then(|json| serde_json::from_str::<ConditionJson>(json).ok())
                .map_or(Condition::None, |cond| self.parse_condition(&cond)),
        }
    }

    fn combine_conditions(&self, conditions: &[ConditionValue]) -> Condition {
        let mut parsed: Vec<Condition> = conditions
            .iter()
            .map(|c| self.resolve_condition(c))
            .filter(|c| *c != Condition::None)
            .collect();
        match parsed.len() {
            0 => Condition::None,
            1 => parsed.remove(0),
            _ => Condition::AllOf(parsed),
        }
    }

    // Only the shapes Pumpkin's loot roller understands are kept, everything else becomes None.
    fn parse_condition(&self, cond: &ConditionJson) -> Condition {
        match cond.condition.as_str() {
            "minecraft:survives_explosion" => Condition::SurvivesExplosion,
            "minecraft:killed_by_player" => Condition::KilledByPlayer,
            "minecraft:random_chance" => Condition::RandomChance {
                chance: cond
                    .chance
                    .or_else(|| cond.chances.as_ref().and_then(|c| c.first().copied()))
                    .unwrap_or(0.0),
            },
            "minecraft:random_chance_with_enchanted_bonus" => {
                let unenchanted_chance = cond.unenchanted_chance.unwrap_or(0.0);
                let (enchanted_chance_base, enchanted_chance_per_level_above_first) =
                    match &cond.enchanted_chance {
                        Some(EnchantedChanceJson::Linear {
                            base,
                            per_level_above_first,
                            ..
                        }) => (*base, *per_level_above_first),
                        Some(EnchantedChanceJson::Constant(c)) => (*c, 0.0),
                        None => (unenchanted_chance, 0.0),
                    };
                Condition::RandomChanceWithEnchantedBonus {
                    unenchanted_chance,
                    enchanted_chance_base,
                    enchanted_chance_per_level_above_first,
                }
            }
            "minecraft:table_bonus" => match &cond.chances {
                Some(chances) if !chances.is_empty() => Condition::TableBonus {
                    chances: chances.clone().into_boxed_slice(),
                },
                _ => Condition::None,
            },
            "minecraft:all_of" => cond
                .terms
                .as_ref()
                .map_or(Condition::None, |terms| self.combine_conditions(terms)),
            "minecraft:match_tool" => {
                let Some(predicate) = &cond.predicate else {
                    return Condition::None;
                };
                let is_shears = match &predicate.items {
                    Some(serde_json::Value::String(s)) => s.contains("shears"),
                    Some(serde_json::Value::Array(items)) => items
                        .iter()
                        .any(|v| v.as_str().is_some_and(|s| s.contains("shears"))),
                    _ => false,
                };
                if is_shears {
                    Condition::Shears
                } else if predicate
                    .predicates
                    .as_ref()
                    .is_some_and(|p| p.to_string().contains("silk_touch"))
                {
                    Condition::SilkTouch
                } else {
                    Condition::None
                }
            }
            "minecraft:any_of" => {
                let Some(terms) = &cond.terms else {
                    return Condition::None;
                };
                let resolved: Vec<Condition> =
                    terms.iter().map(|t| self.resolve_condition(t)).collect();
                let has_silk = resolved.contains(&Condition::SilkTouch);
                let has_shears = resolved.contains(&Condition::Shears);
                match (has_silk, has_shears) {
                    (true, true) => Condition::SilkTouchOrShears,
                    (true, false) => Condition::SilkTouch,
                    (false, true) => Condition::Shears,
                    (false, false) => Condition::None,
                }
            }
            "minecraft:inverted" => match cond.term.as_ref().map(|t| self.resolve_condition(t)) {
                Some(Condition::SilkTouch) => Condition::NoSilkTouch,
                Some(Condition::Shears | Condition::SilkTouchOrShears) => {
                    Condition::NoSilkTouchOrShears
                }
                _ => Condition::None,
            },
            _ => Condition::None,
        }
    }

    fn extract_entries(
        &self,
        entry: &EntryJson,
        inherited: Condition,
        out: &mut Vec<DynamicLootEntry>,
        empty_weight: &mut i32,
        depth: usize,
    ) {
        if depth > MAX_NESTING {
            return;
        }

        let entry_cond = match (inherited, self.condition_of(entry.condition.as_ref())) {
            (Condition::None, cond) | (cond, Condition::None) => cond,
            (first, second) if first == second => first,
            (first, second) => Condition::AllOf(vec![first, second]),
        };

        match entry.entry_type.as_str() {
            "minecraft:empty" => *empty_weight += entry.weight,
            "minecraft:item" => {
                let Some(name) = &entry.name else {
                    return;
                };
                let (min_count, max_count) = entry
                    .functions
                    .iter()
                    .find(|f| f.function == "minecraft:set_count")
                    .and_then(|f| f.count.as_ref())
                    .map_or((1, 1), |c| (c.min(), c.max()));
                out.push(DynamicLootEntry {
                    item: name.clone(),
                    weight: entry.weight,
                    min_count,
                    max_count,
                    condition: entry_cond,
                    bonus_formula: entry.functions.iter().find_map(FunctionJson::bonus_formula),
                });
            }
            "minecraft:tag" => self.push_tag_items(entry, &entry_cond, out),
            "minecraft:loot_table" => {
                let nested = match &entry.value {
                    Some(TableValue::Inline(table)) => Some(table.clone()),
                    Some(TableValue::Reference(id)) => self.nested_table(id),
                    None => entry.name.as_deref().and_then(|id| self.nested_table(id)),
                };
                for pool in nested.iter().flat_map(|t| &t.pools) {
                    let pool_cond = match self.condition_of(pool.condition.as_ref()) {
                        Condition::None => entry_cond.clone(),
                        cond => cond,
                    };
                    for child in &pool.entries {
                        self.extract_entries(
                            child,
                            pool_cond.clone(),
                            out,
                            empty_weight,
                            depth + 1,
                        );
                    }
                }
            }
            "minecraft:alternatives" => {
                let mut saw_silk = false;
                let mut saw_shears = false;
                for child in &entry.children {
                    // later children only apply when the earlier silk touch or shears ones didn't
                    let cond = match self.condition_of(child.condition.as_ref()) {
                        Condition::SilkTouch => {
                            saw_silk = true;
                            Condition::SilkTouch
                        }
                        Condition::Shears => {
                            saw_shears = true;
                            Condition::Shears
                        }
                        Condition::SilkTouchOrShears => {
                            saw_silk = true;
                            saw_shears = true;
                            Condition::SilkTouchOrShears
                        }
                        _ if saw_shears => Condition::NoSilkTouchOrShears,
                        _ if saw_silk => Condition::NoSilkTouch,
                        _ => entry_cond.clone(),
                    };
                    self.extract_entries(child, cond, out, empty_weight, depth + 1);
                }
            }
            "minecraft:sequence" | "minecraft:group" => {
                for child in &entry.children {
                    self.extract_entries(child, entry_cond.clone(), out, empty_weight, depth + 1);
                }
            }
            _ => {}
        }
    }

    // a tag entry turns into one entry per item in the tag
    fn push_tag_items(&self, entry: &EntryJson, cond: &Condition, out: &mut Vec<DynamicLootEntry>) {
        let tag = entry
            .items
            .as_deref()
            .or(entry.name.as_deref())
            .or(match &entry.value {
                Some(TableValue::Reference(r)) => Some(r.as_str()),
                _ => None,
            });
        let Some(tag) = tag.map(|t| t.strip_prefix('#').unwrap_or(t)) else {
            return;
        };
        let Some(tag) = Self::lookup(&self.item_tags, tag)
            .and_then(|json| serde_json::from_str::<TagJson>(json).ok())
        else {
            return;
        };
        out.extend(tag.values.into_iter().map(|item| DynamicLootEntry {
            item,
            weight: entry.weight,
            min_count: 1,
            max_count: 1,
            condition: cond.clone(),
            bonus_formula: None,
        }));
    }

    fn nested_table(&self, id: &str) -> Option<TableJson> {
        Self::lookup(&self.loot_tables, id).and_then(|json| serde_json::from_str(json).ok())
    }
}

/// Encodes a converted table for storage in the pack.
pub fn encode(table: &DynamicLootTable) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(table)
}

static CACHE: LazyLock<DashMap<String, Option<Arc<DynamicLootTable>>>> =
    LazyLock::new(DashMap::new);

/// Returns a vanilla loot table from the installed pack, decoding it on first use.
///
/// `key` may be bare (`blocks/stone`) or namespaced (`minecraft:blocks/stone`).
#[must_use]
pub fn get(key: &str) -> Option<Arc<DynamicLootTable>> {
    let key = if key.contains(':') {
        key.to_owned()
    } else {
        format!("minecraft:{key}")
    };
    if let Some(cached) = CACHE.get(&key) {
        return cached.clone();
    }
    let table = match read(&key) {
        Ok(table) => table.map(Arc::new),
        Err(e) => {
            tracing::error!("Failed to read loot table '{key}' from the vanilla pack: {e}");
            None
        }
    };
    CACHE.insert(key, table.clone());
    table
}

fn read(key: &str) -> Result<Option<DynamicLootTable>, PackError> {
    let Some(pack) = installed() else {
        return Ok(None);
    };
    let Some(bytes) = pack.read(ResourceKind::LootTable, key)? else {
        return Ok(None);
    };
    postcard::from_bytes(&bytes)
        .map(Some)
        .map_err(|_| PackError::CorruptEntry(key.to_owned()))
}

/// Ids of all vanilla loot tables in the installed pack.
#[must_use]
pub fn names() -> Vec<String> {
    let Some(pack) = installed() else {
        return Vec::new();
    };
    match pack.keys(ResourceKind::LootTable) {
        Ok(keys) => keys.map(str::to_owned).collect(),
        Err(e) => {
            tracing::error!("Failed to list vanilla loot tables: {e}");
            Vec::new()
        }
    }
}

#[derive(Deserialize)]
struct TagJson {
    values: Vec<String>,
}

#[derive(Deserialize, Clone)]
struct TableJson {
    #[serde(default)]
    pools: Vec<PoolJson>,
}

#[derive(Deserialize, Clone)]
struct PoolJson {
    #[serde(default)]
    entries: Vec<EntryJson>,
    #[serde(default = "RangeJson::one")]
    rolls: RangeJson,
    #[serde(default)]
    condition: Option<ConditionValue>,
}

#[derive(Deserialize, Clone)]
struct EntryJson {
    #[serde(rename = "type")]
    entry_type: String,
    name: Option<String>,
    // tag entries name their tag here
    #[serde(default)]
    items: Option<String>,
    #[serde(default)]
    value: Option<TableValue>,
    #[serde(default = "default_weight")]
    weight: i32,
    #[serde(rename = "modifier", default, deserialize_with = "one_or_many")]
    functions: Vec<FunctionJson>,
    #[serde(default)]
    condition: Option<ConditionValue>,
    #[serde(default)]
    children: Vec<Self>,
}

const fn default_weight() -> i32 {
    1
}

#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum TableValue {
    Reference(String),
    Inline(TableJson),
}

/// A number or number provider, used for `rolls` and `set_count`.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum RangeJson {
    Constant(f32),
    Provider {
        // required so plain objects without a type don't match
        #[serde(rename = "type")]
        _kind: String,
        #[serde(default)]
        min: f32,
        #[serde(default)]
        max: f32,
    },
}

impl RangeJson {
    const fn one() -> Self {
        Self::Constant(1.0)
    }

    const fn min(&self) -> i32 {
        match self {
            Self::Constant(v) | Self::Provider { min: v, .. } => v.round() as i32,
        }
    }

    const fn max(&self) -> i32 {
        match self {
            Self::Constant(v) | Self::Provider { max: v, .. } => v.round() as i32,
        }
    }
}

#[derive(Deserialize, Clone)]
struct FunctionJson {
    #[serde(rename = "type")]
    function: String,
    #[serde(default)]
    formula: Option<String>,
    #[serde(default)]
    parameters: Option<BonusParametersJson>,
    count: Option<RangeJson>,
}

impl FunctionJson {
    fn bonus_formula(&self) -> Option<LootBonusFormula> {
        let params = self.parameters.as_ref();
        match self.function.as_str() {
            "minecraft:apply_bonus" => match self.formula.as_deref() {
                Some("minecraft:ore_drops") => Some(LootBonusFormula::OreDrops),
                Some("minecraft:uniform_bonus_count") => Some(LootBonusFormula::UniformBonusCount(
                    params.and_then(|p| p.bonus_multiplier).unwrap_or(1),
                )),
                Some("minecraft:binomial_with_bonus_count") => {
                    Some(LootBonusFormula::BinomialWithBonusCount {
                        extra: params.and_then(|p| p.extra).unwrap_or(0),
                        probability: params.and_then(|p| p.probability).unwrap_or(0.0),
                    })
                }
                _ => None,
            },
            "minecraft:enchanted_count_increase" => Some(LootBonusFormula::UniformBonusCount(
                self.count.as_ref().map_or(1, RangeJson::max),
            )),
            _ => None,
        }
    }
}

#[derive(Deserialize, Clone)]
struct BonusParametersJson {
    #[serde(rename = "bonusMultiplier", default)]
    bonus_multiplier: Option<i32>,
    #[serde(default)]
    extra: Option<i32>,
    #[serde(default)]
    probability: Option<f32>,
}

/// A condition is either inline or the id of a predicate file.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum ConditionValue {
    Reference(String),
    Inline(Box<ConditionJson>),
}

#[derive(Deserialize, Clone)]
struct ConditionJson {
    #[serde(rename = "type", default)]
    condition: String,
    #[serde(default)]
    chance: Option<f32>,
    #[serde(default)]
    unenchanted_chance: Option<f32>,
    #[serde(default)]
    enchanted_chance: Option<EnchantedChanceJson>,
    #[serde(default)]
    chances: Option<Vec<f32>>,
    #[serde(default)]
    predicate: Option<ToolPredicateJson>,
    #[serde(default)]
    term: Option<ConditionValue>,
    #[serde(default)]
    terms: Option<Vec<ConditionValue>>,
}

#[derive(Deserialize, Clone)]
struct ToolPredicateJson {
    #[serde(default)]
    items: Option<serde_json::Value>,
    #[serde(default)]
    predicates: Option<serde_json::Value>,
}

#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum EnchantedChanceJson {
    Constant(f32),
    Linear {
        #[serde(rename = "type")]
        _kind: String,
        base: f32,
        #[serde(default)]
        per_level_above_first: f32,
    },
}

/// Item modifiers hold either a single entry or a list of them.
fn one_or_many<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    // `Many` goes first, a json value would also match `One`
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany<T> {
        Many(Vec<T>),
        One(T),
    }

    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::Many(values) => values,
        OneOrMany::One(value) => vec![value],
    })
}
