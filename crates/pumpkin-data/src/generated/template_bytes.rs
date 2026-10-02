/* This file is generated. Do not edit manually. */
#[allow(clippy::too_many_lines)]
#[allow(clippy::match_same_arms)]
#[allow(clippy::missing_const_for_fn)]
#[allow(clippy::match_single_binding)]
#[must_use]
pub fn get_template_bytes(path: &str) -> Option<&'static [u8]> {
    match path {
        "pumpkin:creeper_should_run_from_cat" => Some(include_bytes!(
            "../../../../assets/tests/datapacks/pumpkin-unit-test-example/data/pumpkin/structure/creeper_should_run_from_cat.nbt"
        )),
        "pumpkin:summon_cat" => Some(include_bytes!(
            "../../../../assets/tests/datapacks/pumpkin-unit-test/data/pumpkin/structure/summon_cat.nbt"
        )),
        _ => None,
    }
}
#[must_use]
#[allow(clippy::too_many_lines, clippy::large_stack_arrays)]
pub const fn all_template_names() -> &'static [&'static str] {
    &["pumpkin:creeper_should_run_from_cat", "pumpkin:summon_cat"]
}
#[must_use]
#[allow(clippy::too_many_lines, clippy::large_stack_arrays)]
pub const fn all_embedded_datapack_names() -> &'static [&'static str] {
    &["vanilla", "pumpkin-unit-test", "pumpkin-unit-test-example"]
}
