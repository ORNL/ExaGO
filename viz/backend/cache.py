"""Case preparation and caching."""

import hashlib
import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
from shapely import STRtree, from_wkt

from .data import load_case, reference_tables
from .prepared import MapReferences, prepare_map


@dataclass
class PreparedCase:
    case_id: str
    response: str
    connection: object = None
    chatgrid_error: str | None = None
    query_lock: object = field(default_factory=threading.Lock)


class CaseCache:
    """Cache for a case object."""

    def __init__(self, data_path, reference_data_path, assets_path, *, chatgrid=False):
        self.data_path = Path(data_path)
        self.reference_data_path = Path(reference_data_path)
        self.assets_path = Path(assets_path)
        self.chatgrid = chatgrid
        self._lock = threading.Lock()
        self._cases = {}
        self._resources = {}

    def _once(self, registry, key, prepare):
        with self._lock:
            future = registry.get(key)
            owner = future is None
            if owner:
                future = registry[key] = Future()
        if owner:
            try:
                future.set_result(prepare())
            except Exception as error:
                future.set_exception(error)
        return future.result()

    def _map_references(self):
        def read():
            # NOTE: some non-UTF-8 county names are present inside the geojson
            with (self.assets_path / "data/counties.json").open(
                encoding="utf-8", errors="replace"
            ) as source:
                return MapReferences(json.load(source))

        return self._once(self._resources, "map", read)

    def _database_references(self):
        def read():
            tables = reference_tables(self.data_path, self.reference_data_path)
            return tables, STRtree(from_wkt(tables["counties"]["WKT"].to_numpy()))

        return self._once(self._resources, "database", read)

    def _install_spatial(self):
        def install():
            with duckdb.connect(":memory:") as connection:
                connection.execute("INSTALL spatial")

        self._once(self._resources, "spatial", install)

    def prepare(self, data):
        canonical = json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        case_id = hashlib.sha256(canonical.encode()).hexdigest()

        def build():
            prepared = prepare_map(data, self._map_references())
            entry = PreparedCase(case_id, "")
            if self.chatgrid:
                connection = None
                try:
                    references, county_index = self._database_references()
                    connection = self._open_database()
                    load_case(connection, data, references, county_index)
                    entry.connection = connection
                except Exception as error:
                    if connection is not None:
                        connection.close()
                    entry.chatgrid_error = str(error)
            entry.response = json.dumps(
                {
                    "case_id": case_id,
                    "prepared": prepared,
                    "chatgrid": self.chatgrid_status(entry),
                },
                allow_nan=False,
                separators=(",", ":"),
            )
            return entry

        return self._once(self._cases, case_id, build)

    def _open_database(self):
        self._install_spatial()
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("LOAD spatial")
        except Exception:
            connection.close()
            raise
        return connection

    def get(self, case_id):
        with self._lock:
            future = self._cases.get(case_id)
        if future is None:
            raise KeyError(
                "Unknown case identifier. Reload the dataset after restarting the server."
            )
        return future.result()

    def chatgrid_status(self, entry):
        if not self.chatgrid:
            return {"status": "disabled"}
        if entry.chatgrid_error is not None:
            return {"status": "failed", "error": entry.chatgrid_error}
        return {"status": "ready"}

    def close(self):
        with self._lock:
            futures = list(self._cases.values())
        for future in futures:
            if future.done() and future.exception() is None:
                entry = future.result()
                with entry.query_lock:
                    if entry.connection is not None:
                        entry.connection.close()
