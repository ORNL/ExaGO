"""Tests for the Modification Engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentigrid.engine import process_commands
from agentigrid.engine.commands import (
    ScaleAllLoads,
    ScaleLoad,
    SetBranchStatus,
    SetBusVLimits,
    SetGenDispatch,
    SetGenStatus,
    SetGenVoltage,
    SetLoad,
    parse_command,
)
from agentigrid.engine.modifier import apply_modifications
from agentigrid.engine.schema_description import command_schema_text
from agentigrid.engine.validation import validate_command
from agentigrid.parsers import parse_matpower

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ACTIVSG200 = DATA_DIR / "case_ACTIVSg200.m"
_has_test_file = ACTIVSG200.exists()


@pytest.fixture(scope="module")
def net():
    if not _has_test_file:
        pytest.skip("case_ACTIVSg200.m not in data/")
    return parse_matpower(ACTIVSG200)


# ===========================================================================
# Command parsing tests
# ===========================================================================

class TestParseCommand:

    def test_parse_set_load(self):
        cmd = parse_command({"action": "set_load", "bus": 10, "Pd": 50.0})
        assert isinstance(cmd, SetLoad)
        assert cmd.bus == 10
        assert cmd.Pd == 50.0
        assert cmd.Qd is None

    def test_parse_scale_all_loads(self):
        cmd = parse_command({"action": "scale_all_loads", "factor": 1.2})
        assert isinstance(cmd, ScaleAllLoads)
        assert cmd.factor == 1.2

    def test_parse_scale_load_area(self):
        cmd = parse_command({"action": "scale_load", "factor": 0.8, "area": 4})
        assert isinstance(cmd, ScaleLoad)
        assert cmd.area == 4

    def test_parse_set_gen_status(self):
        cmd = parse_command({"action": "set_gen_status", "bus": 189, "status": 0})
        assert isinstance(cmd, SetGenStatus)

    def test_parse_set_gen_dispatch(self):
        cmd = parse_command({"action": "set_gen_dispatch", "bus": 189, "Pg": 400.0})
        assert isinstance(cmd, SetGenDispatch)

    def test_parse_set_branch_status(self):
        cmd = parse_command({"action": "set_branch_status", "fbus": 2, "tbus": 1, "status": 0})
        assert isinstance(cmd, SetBranchStatus)

    def test_parse_set_bus_vlimits(self):
        cmd = parse_command({"action": "set_bus_vlimits", "bus": 10, "Vmin": 0.95})
        assert isinstance(cmd, SetBusVLimits)

    def test_unknown_action(self):
        with pytest.raises(ValueError, match="Unknown action"):
            parse_command({"action": "destroy_grid"})

    def test_missing_required(self):
        with pytest.raises(ValueError, match="missing required"):
            parse_command({"action": "set_load"})

    def test_missing_action_key(self):
        with pytest.raises(ValueError, match="missing 'action'"):
            parse_command({"bus": 10})


# ===========================================================================
# Validation tests
# ===========================================================================

class TestValidation:

    def test_valid_set_load(self, net):
        cmd = SetLoad(bus=10, Pd=50.0)
        result = validate_command(cmd, net)
        assert result.valid

    def test_nonexistent_bus(self, net):
        cmd = SetLoad(bus=999, Pd=50.0)
        result = validate_command(cmd, net)
        assert not result.valid
        assert any("999" in e for e in result.errors)

    def test_nonexistent_gen(self, net):
        # Bus 10 has load but no generator
        cmd = SetGenStatus(bus=10, status=0)
        result = validate_command(cmd, net)
        assert not result.valid
        assert any("No generator" in e for e in result.errors)

    def test_pg_out_of_bounds(self, net):
        # Bus 189 gen has Pmax=569.15
        cmd = SetGenDispatch(bus=189, Pg=9999.0)
        result = validate_command(cmd, net)
        assert not result.valid
        assert any("outside bounds" in e for e in result.errors)

    def test_scale_factor_zero(self, net):
        cmd = ScaleAllLoads(factor=0)
        result = validate_command(cmd, net)
        assert not result.valid

    def test_scale_factor_negative(self, net):
        cmd = ScaleAllLoads(factor=-1.0)
        result = validate_command(cmd, net)
        assert not result.valid

    def test_large_scale_factor_warns(self, net):
        cmd = ScaleAllLoads(factor=5.0)
        result = validate_command(cmd, net)
        assert result.valid  # not an error
        assert len(result.warnings) > 0

    def test_already_offline_gen_warns(self, net):
        # Bus 78 has an offline gen (status=0 in the file)
        cmd = SetGenStatus(bus=78, status=0)
        result = validate_command(cmd, net)
        assert result.valid
        assert any("already offline" in w for w in result.warnings)

    def test_valid_branch(self, net):
        cmd = SetBranchStatus(fbus=2, tbus=1, status=0)
        result = validate_command(cmd, net)
        assert result.valid

    def test_nonexistent_branch(self, net):
        cmd = SetBranchStatus(fbus=1, tbus=200, status=0)
        result = validate_command(cmd, net)
        assert not result.valid

    def test_bus_vlimits_invalid(self, net):
        cmd = SetBusVLimits(bus=1, Vmin=1.1, Vmax=0.9)
        result = validate_command(cmd, net)
        assert not result.valid

    def test_slack_bus_dispatch_rejected(self, net):
        cmd = SetGenDispatch(bus=189, Pg=400.0)
        result = validate_command(cmd, net)
        assert not result.valid
        assert any("reference/slack bus" in e for e in result.errors)


class TestModifications:

    def test_slack_bus_dispatch_skipped(self, net):
        bus189_pg_before = next(g.Pg for g in net.generators if g.bus == 189)
        cmd = SetGenDispatch(bus=189, Pg=500.0)
        modified, report = apply_modifications(net, [cmd])
        bus189_pg_after = next(g.Pg for g in modified.generators if g.bus == 189)
        assert bus189_pg_after == bus189_pg_before
        assert len(report.skipped) == 1
        assert any("reference/slack bus" in r[1][0] for r in report.skipped)

    def test_set_load(self, net):
        cmds = [SetLoad(bus=10, Pd=99.0, Qd=33.0)]
        modified, report = apply_modifications(net, cmds)
        bus10 = next(b for b in modified.buses if b.bus_i == 10)
        assert bus10.Pd == 99.0
        assert bus10.Qd == 33.0
        assert len(report.applied) == 1

    def test_set_load_other_buses_unchanged(self, net):
        orig_bus1_pd = next(b for b in net.buses if b.bus_i == 1).Pd
        cmds = [SetLoad(bus=10, Pd=99.0)]
        modified, _ = apply_modifications(net, cmds)
        bus1 = next(b for b in modified.buses if b.bus_i == 1)
        assert bus1.Pd == orig_bus1_pd

    def test_scale_all_loads(self, net):
        factor = 1.5
        orig_pds = {b.bus_i: b.Pd for b in net.buses}
        cmds = [ScaleAllLoads(factor=factor)]
        modified, _ = apply_modifications(net, cmds)
        for b in modified.buses:
            assert abs(b.Pd - orig_pds[b.bus_i] * factor) < 1e-6

    def test_scale_load_area(self, net):
        target_area = 1
        factor = 2.0
        area_buses = {b.bus_i for b in net.buses if b.area == target_area}
        orig = {b.bus_i: b.Pd for b in net.buses}

        cmds = [ScaleLoad(factor=factor, area=target_area)]
        modified, _ = apply_modifications(net, cmds)

        for b in modified.buses:
            if b.bus_i in area_buses:
                assert abs(b.Pd - orig[b.bus_i] * factor) < 1e-6
            else:
                assert abs(b.Pd - orig[b.bus_i]) < 1e-6

    def test_set_gen_status(self, net):
        cmds = [SetGenStatus(bus=189, status=0)]
        modified, report = apply_modifications(net, cmds)
        gen189 = next(g for g in modified.generators if g.bus == 189)
        assert gen189.status == 0
        assert len(report.applied) == 1

    def test_set_gen_dispatch(self, net):
        cmds = [SetGenDispatch(bus=125, Pg=80.0)]
        modified, _ = apply_modifications(net, cmds)
        gen125 = next(g for g in modified.generators if g.bus == 125)
        assert gen125.Pg == 80.0

    def test_set_branch_status(self, net):
        cmds = [SetBranchStatus(fbus=2, tbus=1, status=0)]
        modified, _ = apply_modifications(net, cmds)
        br = next(br for br in modified.branches if br.fbus == 2 and br.tbus == 1)
        assert br.status == 0

    def test_set_bus_vlimits(self, net):
        cmds = [SetBusVLimits(bus=1, Vmin=0.95, Vmax=1.05)]
        modified, _ = apply_modifications(net, cmds)
        bus1 = next(b for b in modified.buses if b.bus_i == 1)
        assert bus1.Vmin == 0.95
        assert bus1.Vmax == 1.05

    def test_deep_copy_original_unchanged(self, net):
        orig_pd = next(b for b in net.buses if b.bus_i == 10).Pd
        cmds = [SetLoad(bus=10, Pd=9999.0)]
        apply_modifications(net, cmds)
        assert next(b for b in net.buses if b.bus_i == 10).Pd == orig_pd


# ===========================================================================
# Full pipeline test
# ===========================================================================

class TestProcessCommands:

    def test_mixed_valid_and_invalid(self, net):
        raw = [
            {"action": "set_load", "bus": 10, "Pd": 50.0},
            {"action": "set_load", "bus": 999, "Pd": 50.0},  # invalid bus
            {"action": "scale_all_loads", "factor": 1.1},
            {"action": "unknown_action"},  # parse error
        ]
        modified, report = process_commands(net, raw)

        # 2 applied (set_load on bus 10, scale_all_loads)
        assert len(report.applied) == 2
        # 2 skipped (bus 999 + unknown action)
        assert len(report.skipped) == 2


# ===========================================================================
# Schema description test
# ===========================================================================

class TestSchemaDescription:

    def test_schema_contains_all_commands(self):
        text = command_schema_text()
        for action in [
            "set_load", "scale_load", "scale_all_loads", "set_gen_status",
            "set_gen_dispatch", "set_gen_voltage", "set_branch_status",
            "set_branch_rate", "set_cost_coeffs", "set_bus_vlimits",
        ]:
            assert action in text


# ===========================================================================
# set_gen_voltage OPFLOW warning tests
# ===========================================================================

class TestSetGenVoltageWarning:

    @pytest.fixture
    def net(self):
        if not _has_test_file:
            pytest.skip("ACTIVSg200 not found")
        return parse_matpower(ACTIVSG200)

    def test_no_warning_without_application(self, net):
        """No warning when application is not specified."""
        cmd = SetGenVoltage(bus=189, Vg=1.02)
        _, report = apply_modifications(net, [cmd])
        assert not any("set_gen_voltage" in w.lower() for w in report.warnings)

    def test_no_warning_for_pflow(self, net):
        """No warning when application is pflow (set_gen_voltage is valid there)."""
        cmd = SetGenVoltage(bus=189, Vg=1.02)
        _, report = apply_modifications(net, [cmd], application="pflow")
        assert not any("override" in w for w in report.warnings)

    def test_warning_for_opflow(self, net):
        """Warning emitted when application is opflow."""
        cmd = SetGenVoltage(bus=189, Vg=1.02)
        _, report = apply_modifications(net, [cmd], application="opflow")
        assert any("OPFLOW" in w for w in report.warnings)

    def test_command_still_applied_despite_warning(self, net):
        """set_gen_voltage is still applied (Vg updated) even when warning fires."""
        cmd = SetGenVoltage(bus=189, Vg=0.99)
        modified, report = apply_modifications(net, [cmd], application="opflow")
        assert len(report.applied) == 1
        gen189 = next(g for g in modified.generators if g.bus == 189)
        assert abs(gen189.Vg - 0.99) < 1e-6

    def test_schema_contains_warning(self):
        """The command schema text warns about OPFLOW for set_gen_voltage."""
        text = command_schema_text()
        assert "WARNING" in text
        assert "set_bus_vlimits" in text  # points to the correct alternative
