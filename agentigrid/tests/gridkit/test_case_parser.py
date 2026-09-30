"""Tests for the GridKit case reader and element map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentigrid.gridkit_parsers.case_parser import (
    CaseFileError,
    build_element_map,
    column_owner,
    column_owners,
    create_element_map,
    devices_at_bus,
    fault_element_id,
    load_case,
    load_element_map,
)

ROOT = Path(__file__).resolve().parents[2]
IEEE39 = ROOT / "data" / "gridkit" / "examples" / "IEEE39.case.json"


def _small_case() -> dict:
    """Three buses; machine at bus 2 with exciter, stabilizer and governor.

    The governor and stabilizer have only signal ports. The stabilizer reaches
    the bus through the exciter (two signal hops). The fault at bus 3 comes
    first in the file, so it is element 0. Bus 2 is named "Two".
    """
    return {
        "header": {"case_name": "Small"},
        "buses": [
            {"number": 1, "class": "BusInfinite", "name": "1"},
            {"number": 2, "class": "Bus", "name": "Two"},
            {"number": 3, "class": "Bus", "name": "3"},
        ],
        "signals": [{"signal_id": i} for i in range(5)],
        "devices": [
            {"class": "Branch", "id": "br_1_2", "ports": {"bus1": 1, "bus2": 2}},
            {"class": "Branch", "id": "br_2_3", "ports": {"bus1": 2, "bus2": 3}},
            {"class": "Genrou", "id": "g2", "ports": {"bus": 2, "speed": 0, "pmech": 1, "efd": 2}},
            {"class": "Tgov1", "id": "gov2", "ports": {"speed": 0, "pmech": 1}},
            {"class": "Ieeet1", "id": "exc2", "ports": {"bus": 2, "efd": 2, "speed": 0, "vs": 3}},
            {"class": "Ieeest", "id": "pss2", "ports": {"input": 4, "output": 3}},
            {"class": "BusFault", "id": "f3", "ports": {"bus": 3}},
            {"class": "LoadZIP", "id": "ld3", "ports": {"bus": 3}},
            {"class": "BusFault", "id": "f1", "ports": {"bus": 1}},
        ],
    }


@pytest.fixture
def emap() -> dict:
    return build_element_map(_small_case(), "small.case.json")


class TestElementMap:
    def test_counts(self, emap):
        assert emap["case_name"] == "Small"
        assert emap["counts"]["buses"] == 3
        assert emap["counts"]["devices"] == 9
        assert emap["counts"]["by_class"]["BusFault"] == 2

    def test_fault_element_ids_follow_file_order(self, emap):
        assert emap["bus_faults"] == [
            {"element_id": 0, "id": "f3", "bus": 3},
            {"element_id": 1, "id": "f1", "bus": 1},
        ]
        assert fault_element_id(emap, 3) == 0
        assert fault_element_id(emap, 1) == 1
        assert fault_element_id(emap, 2) is None

    def test_signal_only_devices_get_a_bus(self, emap):
        by_id = {d["id"]: d for d in emap["devices"]}
        assert by_id["gov2"]["buses"] == [2] and by_id["gov2"]["via"] == "signal"
        assert by_id["pss2"]["buses"] == [2] and by_id["pss2"]["via"] == "signal"
        assert by_id["g2"]["via"] == "port"

    def test_devices_at_bus(self, emap):
        ids = {d["id"] for d in devices_at_bus(emap, 2)}
        assert ids == {"br_1_2", "br_2_3", "g2", "gov2", "exc2", "pss2"}
        bus2 = next(b for b in emap["buses"] if b["bus"] == 2)
        assert set(bus2["devices"]) == ids

    def test_machines(self, emap):
        assert emap["machines"] == [{"id": "g2", "class": "Genrou", "bus": 2}]

    def test_column_owner_uses_bus_name(self, emap):
        assert column_owner(emap, "Bus_Two_Vm") == {
            "kind": "bus", "bus": 2, "class": "Bus", "id": None, "variable": "Vm",
        }
        assert column_owner(emap, "Genrou_g2_omega")["bus"] == 2
        assert column_owner(emap, "t") is None
        assert column_owner(emap, "Genrou_unknown_omega") is None

    def test_column_owner_label_with_underscores(self):
        case = _small_case()
        case["devices"][2]["id"] = "30_1_genrou"
        owners = column_owners(build_element_map(case), ["t", "Genrou_30_1_genrou_omega"])
        assert owners["t"] is None
        assert owners["Genrou_30_1_genrou_omega"]["id"] == "30_1_genrou"
        assert owners["Genrou_30_1_genrou_omega"]["variable"] == "omega"

    def test_create_saves_and_reloads(self, tmp_path):
        case_file = tmp_path / "small.case.json"
        case_file.write_text(json.dumps(_small_case()))
        emap, path = create_element_map(case_file, tmp_path / "session")
        assert path == tmp_path / "session" / "element_map.json"
        assert load_element_map(path) == emap
        assert load_element_map(tmp_path / "session") == emap


class TestLoadCase:
    def test_missing_file(self, tmp_path):
        with pytest.raises(CaseFileError, match="not found"):
            load_case(tmp_path / "none.case.json")

    def test_not_json(self, tmp_path):
        bad = tmp_path / "bad.case.json"
        bad.write_text("{")
        with pytest.raises(CaseFileError, match="not valid JSON"):
            load_case(bad)

    def test_missing_devices(self, tmp_path):
        bad = tmp_path / "bad.case.json"
        bad.write_text(json.dumps({"buses": []}))
        with pytest.raises(CaseFileError, match="devices"):
            load_case(bad)


@pytest.mark.skipif(not IEEE39.exists(), reason="data/gridkit/examples/IEEE39.case.json not linked")
class TestIEEE39:
    def test_every_bus_has_a_fault_and_ids_differ_from_bus_numbers(self):
        emap = build_element_map(load_case(IEEE39), IEEE39)
        assert emap["counts"]["buses"] == 39
        assert len(emap["bus_faults"]) == 39
        assert fault_element_id(emap, 16) == 0
        assert fault_element_id(emap, 1) == 1
        assert all(d["buses"] for d in emap["devices"])
