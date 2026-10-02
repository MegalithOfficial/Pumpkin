use serde::{Deserialize, Serialize};

/// Bonus count formulas when tools have fortune or looting enchantments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LootBonusFormula {
    OreDrops,
    UniformBonusCount(i32),
    BinomialWithBonusCount { extra: i32, probability: f32 },
}

/// Conditions required for an entry or pool to be eligible for dynamic loot generation.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum DynamicLootCondition {
    #[default]
    None,
    SilkTouch,
    NoSilkTouch,
    Shears,
    SilkTouchOrShears,
    NoSilkTouchOrShears,
    SurvivesExplosion,
    KilledByPlayer,
    RandomChance {
        chance: f32,
    },
    RandomChanceWithEnchantedBonus {
        unenchanted_chance: f32,
        enchanted_chance_base: f32,
        enchanted_chance_per_level_above_first: f32,
    },
    TableBonus {
        chances: Box<[f32]>,
    },
    AllOf(Vec<Self>),
    AnyOf(Vec<Self>),
    Inverted(Box<Self>),
    EntityOnFire,
    WeatherCheck {
        raining: Option<bool>,
        thundering: Option<bool>,
    },
}

/// A single item entry inside a dynamic loot pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicLootEntry {
    /// Registry name of the item (e.g. `"minecraft:diamond"`).
    pub item: String,
    /// Relative probability weight; higher values are more likely.
    pub weight: i32,
    /// Minimum stack size (inclusive).
    pub min_count: i32,
    /// Maximum stack size (inclusive).
    pub max_count: i32,
    /// Condition required for this entry to be eligible.
    pub condition: DynamicLootCondition,
    /// Bonus formula to apply with fortune / looting (if any).
    pub bonus_formula: Option<LootBonusFormula>,
}

/// One roll pool inside a dynamic loot table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicLootPool {
    /// Item entries eligible for selection each roll.
    pub entries: Vec<DynamicLootEntry>,
    /// Minimum number of roll attempts (inclusive).
    pub min_rolls: i32,
    /// Maximum number of roll attempts (inclusive).
    pub max_rolls: i32,
    /// Weight of the implicit "empty" (no item) outcome per roll.
    pub empty_weight: i32,
    /// Condition required for this entire pool to run.
    pub condition: DynamicLootCondition,
}

/// A complete dynamic loot table consisting of one or more pools.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynamicLootTable {
    /// All pools to roll when generating loot for this table.
    pub pools: Vec<DynamicLootPool>,
}
