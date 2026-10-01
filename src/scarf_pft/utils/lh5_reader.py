"""Read one row-partitioned LH5 table as one logical table.

``VirtualLH5Table`` concatenates one ``struct/table`` pair across source files
with one irregular row-boundary index shared by every column. ``field_mask``
fixes the exposed schema. ``set_column_mask`` compiles predicates once into
physical ``(file_id, local_row)`` arrays reused by later reads.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
from lgdo import types
import lh5.io.datatype as lh5_datatype


Projection = dict[str, "Projection | None"]


@dataclass(frozen=True)
class _SchemaNode:
    """Projected LH5/HDF5 schema node."""

    name: str
    path: str
    relpath: str
    is_dataset: bool
    attrs: dict[str, Any]
    dtype: np.dtype | None = None
    tail_shape: tuple[int, ...] = ()
    children: tuple["_SchemaNode", ...] = ()


@dataclass(frozen=True)
class _SelectionIndex:
    """Filtered ordinal to physical row mapping."""

    file_ids: np.ndarray
    local_rows: np.ndarray

    def __len__(self) -> int:
        return int(self.local_rows.shape[0])


@dataclass(frozen=True)
class _MaskField:
    """One scalar predicate field used during mask compilation."""

    name: str
    path: str
    is_bool: bool


@dataclass(frozen=True)
class _MaskJob:
    """A contiguous source-file group scanned by one mask worker."""

    files: tuple[tuple[int, str, int], ...]
    fields: tuple[_MaskField, ...]
    config: Mapping[str, Any]
    block_rows: int | None
    locking: bool | None
    rdcc_nbytes: int | None
    rdcc_nslots: int | None
    rdcc_w0: float | None
    file_dtype: str
    local_dtype: str


@dataclass(frozen=True)
class _MaskResult:
    """Selected physical rows produced by one mask worker."""

    file_ids: np.ndarray
    local_rows: np.ndarray


@dataclass(frozen=True)
class _ReadTask:
    """Rows read from one source file into output positions."""

    file_id: int
    output: slice | np.ndarray
    n_rows: int
    local_slice: slice | None = None
    local_rows: np.ndarray | None = None
    file_ranges: np.ndarray | None = None
    output_ranges: np.ndarray | None = None
    gather: np.ndarray | None = None


@dataclass(frozen=True)
class _ReadPlan:
    """Physical read plan shared by every requested field."""

    n_rows: int
    tasks: tuple[_ReadTask, ...]


@dataclass
class _OpenSource:
    """Cached source file and dataset handles."""

    h5: h5py.File
    datasets: dict[str, h5py.Dataset] = field(default_factory=dict)


class VirtualLH5Table:
    """Expose one LH5 ``struct/table`` across row-partitioned source files.

    Source files are concatenated in input order. The construction-time
    ``field_mask`` fixes the maximum exposed schema. A per-read mask may reduce
    that schema further. ``set_column_mask`` changes the logical row space once;
    ``clear_column_mask`` restores the full row space.
    """

    def __init__(
        self,
        files: Sequence[str | os.PathLike[str]],
        struct: str,
        table: str,
        field_mask: Sequence[str] | None = None,
        *,
        max_open_files: int | None = None,
        locking: bool | None = None,
        rdcc_nbytes: int | None = None,
        rdcc_nslots: int | None = None,
        rdcc_w0: float | None = None,
    ) -> None:
        if len(files) == 0:
            raise ValueError("files must contain at least one source LH5 file")

        self.files = tuple(os.path.expanduser(os.fspath(f)) for f in files)
        self.struct = struct.strip("/")
        self.table = table.strip("/")
        self.struct_path = f"/{self.struct}"
        self.table_path = f"/{self.struct}/{self.table}"
        self.field_mask = None if field_mask is None else tuple(field_mask)
        self.max_open_files = max_open_files
        self.locking = locking
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.rdcc_w0 = rdcc_w0

        self._projection = _compile_projection(self.field_mask)
        self._schema: _SchemaNode
        self._nodes_by_relpath: dict[str, _SchemaNode] = {}

        self._selection: _SelectionIndex | None = None

        self._cache_pid: int | None = None
        self._open_sources: OrderedDict[int, _OpenSource] = OrderedDict()

        first_count = self._discover_schema()
        self._canonical_leaf = _first_dataset(self._schema)
        self._row_counts = self._read_row_counts(self._canonical_leaf.path, first_count)
        self._row_stops = np.cumsum(self._row_counts, dtype=np.int64)
        self._row_starts = self._row_stops - self._row_counts
        self._n_rows = int(self._row_stops[-1])

        self._row_counts.flags.writeable = False
        self._row_starts.flags.writeable = False
        self._row_stops.flags.writeable = False

    def __len__(self) -> int:
        if self._selection is None:
            return self._n_rows
        return len(self._selection)

    @property
    def n_base_rows(self) -> int:
        """Number of rows in the unmasked global row space."""
        return self._n_rows

    @property
    def masked(self) -> bool:
        """Whether a compiled row selection is active."""
        return self._selection is not None

    @property
    def row_counts(self) -> np.ndarray:
        """Read-only per-file row counts."""
        return self._row_counts

    @property
    def row_starts(self) -> np.ndarray:
        """Read-only global start row for every source file."""
        return self._row_starts

    @property
    def row_stops(self) -> np.ndarray:
        """Read-only global stop row for every source file."""
        return self._row_stops

    @property
    def selected_file_ids(self) -> np.ndarray | None:
        """Read-only filtered-ordinal to source-file mapping."""
        if self._selection is None:
            return None
        return self._selection.file_ids

    @property
    def selected_local_rows(self) -> np.ndarray | None:
        """Read-only filtered-ordinal to local-row mapping."""
        if self._selection is None:
            return None
        return self._selection.local_rows

    def set_column_mask(
        self,
        config: Mapping[str, Any],
        *,
        workers: int = 1,
        block_rows: int | None = None,
    ) -> int:
        """Compile mask predicates and activate the filtered row space.

        Mask fields are reconstructed from ``quality``, ``not_mask`` and
        ``interval_mask``. ``workers`` scans contiguous file groups in spawned
        processes. ``block_rows=None`` processes one source file at a time.
        """
        mask_fields = _mask_fields_from_config(config)
        fields = tuple(_mask_field(self._nodes_by_relpath[name.strip("/")], name) for name in mask_fields)
        file_dtype = _unsigned_dtype(max(0, len(self.files) - 1))
        local_dtype = _unsigned_dtype(max(0, int(self._row_counts.max(initial=0)) - 1))

        groups = _partition_file_groups(self._row_counts, max(1, int(workers)))
        jobs = tuple(
            _MaskJob(
                files=tuple(
                    (file_id, self.files[file_id], int(self._row_counts[file_id]))
                    for file_id in group
                ),
                fields=fields,
                config=config,
                block_rows=block_rows,
                locking=self.locking,
                rdcc_nbytes=self.rdcc_nbytes,
                rdcc_nslots=self.rdcc_nslots,
                rdcc_w0=self.rdcc_w0,
                file_dtype=file_dtype.str,
                local_dtype=local_dtype.str,
            )
            for group in groups
        )

        if len(jobs) == 1:
            results = [_compile_mask_job(jobs[0])]
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
                results = list(pool.map(_compile_mask_job, jobs))

        file_parts = [result.file_ids for result in results if result.file_ids.size]
        row_parts = [result.local_rows for result in results if result.local_rows.size]

        if row_parts:
            file_ids = np.concatenate(file_parts)
            local_rows = np.concatenate(row_parts)
        else:
            file_ids = np.empty(0, dtype=file_dtype)
            local_rows = np.empty(0, dtype=local_dtype)

        file_ids.flags.writeable = False
        local_rows.flags.writeable = False
        self._selection = _SelectionIndex(file_ids=file_ids, local_rows=local_rows)
        return len(self._selection)

    def clear_column_mask(self) -> None:
        """Restore the unfiltered global row space."""
        self._selection = None

    def global_indices(self, idx: Any = None) -> np.ndarray:
        """Map logical row indices to unfiltered global row indices."""
        plan_ids = _logical_indices(idx, len(self))
        if self._selection is None:
            return plan_ids

        file_ids = self._selection.file_ids[plan_ids]
        local_rows = self._selection.local_rows[plan_ids]
        return self._row_starts[file_ids] + local_rows

    def read(
        self,
        idx: Any = None,
        *,
        field_mask: Sequence[str] | None = None,
    ) -> types.Table:
        """Read rows from the active logical row space.

        ``field_mask`` may reduce the construction-time projection for this
        call. Mask predicates are not evaluated in this method.
        """
        projection = _compile_projection(field_mask)
        plan = self._make_read_plan(idx)
        leaves = _selected_leaves(self._schema, projection)
        buffers = {
            node.relpath: np.empty((plan.n_rows, *node.tail_shape), dtype=node.dtype)
            for node in leaves
        }
        self._execute_read_plan(plan, leaves, buffers)
        return self._build_group_node(self._schema, projection, buffers)

    def close(self) -> None:
        """Close cached source-file handles for this process."""
        while self._open_sources:
            _, entry = self._open_sources.popitem(last=False)
            entry.h5.close()
        self._cache_pid = None

    def __getstate__(self) -> dict[str, Any]:
        """Serialize logical state without process-local HDF5 handles."""
        state = self.__dict__.copy()
        state["_cache_pid"] = None
        state["_open_sources"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore logical state with an empty process-local HDF5 cache."""
        self.__dict__.update(state)
        self._cache_pid = None
        self._open_sources = OrderedDict()

    def __enter__(self) -> "VirtualLH5Table":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _discover_schema(self) -> int:
        with self._open_source_file(self.files[0]) as src:
            table_obj = src[self.table_path]
            self._schema = self._discover_node(
                table_obj,
                name=self.table,
                relpath="",
                projection=self._projection,
            )
            canonical = _first_dataset(self._schema)
            first_count = int(src[canonical.path].shape[0])

        self._nodes_by_relpath.clear()
        self._index_nodes(self._schema)
        return first_count

    def _discover_node(
        self,
        obj: h5py.Group | h5py.Dataset,
        *,
        name: str,
        relpath: str,
        projection: Projection | None,
    ) -> _SchemaNode:
        attrs = _read_attrs(obj)

        if isinstance(obj, h5py.Dataset):
            return _SchemaNode(
                name=name,
                path=obj.name,
                relpath=relpath,
                is_dataset=True,
                attrs=attrs,
                dtype=np.dtype(obj.dtype),
                tail_shape=tuple(int(x) for x in obj.shape[1:]),
            )

        datatype = str(attrs.get("datatype", ""))
        declared = _declared_group_fields(obj, datatype)
        chosen = declared if projection is None else [field for field in declared if field in projection]

        children: list[_SchemaNode] = []
        for child_name in chosen:
            subprojection = None if projection is None else projection[child_name]
            child_rel = child_name if relpath == "" else f"{relpath}/{child_name}"
            children.append(
                self._discover_node(
                    obj[child_name],
                    name=child_name,
                    relpath=child_rel,
                    projection=subprojection,
                )
            )

        attrs = _project_group_attrs(attrs, [child.name for child in children])
        return _SchemaNode(
            name=name,
            path=obj.name,
            relpath=relpath,
            is_dataset=False,
            attrs=attrs,
            children=tuple(children),
        )

    def _index_nodes(self, node: _SchemaNode) -> None:
        if node.relpath:
            self._nodes_by_relpath[node.relpath] = node
        for child in node.children:
            self._index_nodes(child)

    def _read_row_counts(self, canonical_dataset_path: str, first_count: int) -> np.ndarray:
        counts = np.empty(len(self.files), dtype=np.int64)
        counts[0] = first_count
        for i, filename in enumerate(self.files[1:], start=1):
            with self._open_source_file(filename) as src:
                counts[i] = src[canonical_dataset_path].shape[0]
        return counts

    def _make_read_plan(self, idx: Any) -> _ReadPlan:
        if self._selection is None:
            return self._make_unmasked_plan(idx)
        return self._make_masked_plan(idx)

    def _make_unmasked_plan(self, idx: Any) -> _ReadPlan:
        if idx is None:
            return self._plan_global_span(0, self._n_rows)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._n_rows)
            if step == 1:
                return self._plan_global_span(start, stop)

        global_rows = _logical_indices(idx, self._n_rows)
        file_ids = np.searchsorted(self._row_stops, global_rows, side="right")
        local_rows = global_rows - self._row_starts[file_ids]
        return self._plan_physical_rows(file_ids, local_rows)

    def _make_masked_plan(self, idx: Any) -> _ReadPlan:
        selection = self._selection
        assert selection is not None

        if idx is None:
            return self._plan_sorted_selection(selection.file_ids, selection.local_rows)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(selection))
            if step == 1:
                return self._plan_sorted_selection(
                    selection.file_ids[start:stop],
                    selection.local_rows[start:stop],
                )

        logical = _logical_indices(idx, len(selection))
        return self._plan_physical_rows(
            selection.file_ids[logical],
            selection.local_rows[logical],
        )

    def _plan_global_span(self, start: int, stop: int) -> _ReadPlan:
        if stop <= start:
            return _ReadPlan(n_rows=0, tasks=())

        first_file = int(np.searchsorted(self._row_stops, start, side="right"))
        last_file = int(np.searchsorted(self._row_stops, stop - 1, side="right"))

        tasks: list[_ReadTask] = []
        out_start = 0
        for file_id in range(first_file, last_file + 1):
            global_start = max(start, int(self._row_starts[file_id]))
            global_stop = min(stop, int(self._row_stops[file_id]))
            local_start = global_start - int(self._row_starts[file_id])
            local_stop = global_stop - int(self._row_starts[file_id])
            n_rows = local_stop - local_start
            if n_rows == 0:
                continue
            tasks.append(
                _ReadTask(
                    file_id=file_id,
                    output=slice(out_start, out_start + n_rows),
                    n_rows=n_rows,
                    local_slice=slice(local_start, local_stop),
                )
            )
            out_start += n_rows

        return _ReadPlan(n_rows=stop - start, tasks=tuple(tasks))

    def _plan_sorted_selection(self, file_ids: np.ndarray, local_rows: np.ndarray) -> _ReadPlan:
        n_rows = int(local_rows.shape[0])
        if n_rows == 0:
            return _ReadPlan(n_rows=0, tasks=())

        breaks = np.flatnonzero(file_ids[1:] != file_ids[:-1]) + 1
        bounds = np.concatenate((np.array([0]), breaks, np.array([n_rows])))
        tasks: list[_ReadTask] = []

        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            a = int(a)
            b = int(b)
            rows = np.asarray(local_rows[a:b], dtype=np.int64)
            tasks.append(
                _task_for_rows(
                    file_id=int(file_ids[a]),
                    rows=rows,
                    output=slice(a, b),
                )
            )

        return _ReadPlan(n_rows=n_rows, tasks=tuple(tasks))

    def _plan_physical_rows(self, file_ids: np.ndarray, local_rows: np.ndarray) -> _ReadPlan:
        file_ids = np.asarray(file_ids)
        local_rows = np.asarray(local_rows)
        n_rows = int(local_rows.shape[0])
        if n_rows == 0:
            return _ReadPlan(n_rows=0, tasks=())

        order = np.argsort(file_ids, kind="stable")
        grouped_files = file_ids[order]
        grouped_rows = local_rows[order]
        breaks = np.flatnonzero(grouped_files[1:] != grouped_files[:-1]) + 1
        bounds = np.concatenate((np.array([0]), breaks, np.array([n_rows])))

        tasks: list[_ReadTask] = []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            a = int(a)
            b = int(b)
            rows = np.asarray(grouped_rows[a:b], dtype=np.int64)
            output = np.asarray(order[a:b], dtype=np.intp)
            tasks.append(
                _task_for_rows(
                    file_id=int(grouped_files[a]),
                    rows=rows,
                    output=output,
                )
            )

        return _ReadPlan(n_rows=n_rows, tasks=tuple(tasks))

    def _execute_read_plan(
        self,
        plan: _ReadPlan,
        leaves: tuple[_SchemaNode, ...],
        buffers: dict[str, np.ndarray],
    ) -> None:
        """Execute one physical plan file-by-file across all requested leaves."""
        for task in plan.tasks:
            source = self._get_source(task.file_id)
            for node in leaves:
                ds = self._get_dataset_from_source(source, node.path)
                _read_task_into(ds, task, buffers[node.relpath], node.tail_shape)

    def _build_group_node(
        self,
        node: _SchemaNode,
        projection: Projection | None,
        buffers: Mapping[str, np.ndarray],
    ) -> Any:
        """Reconstruct the projected LGDO tree from filled leaf buffers."""
        if node.is_dataset:
            nda = _cast_bool_if_needed(buffers[node.relpath], node.attrs)
            return _make_lgdo_array(nda, node.attrs)

        selected_children = (
            node.children
            if projection is None
            else tuple(child for child in node.children if child.name in projection)
        )

        col_dict: dict[str, Any] = {}
        for child in selected_children:
            child_projection = None if projection is None else projection[child.name]
            col_dict[child.name] = self._build_group_node(
                child, child_projection, buffers
            )

        attrs = _project_group_attrs(node.attrs, [child.name for child in selected_children])
        datatype = str(attrs.get("datatype", ""))

        if datatype.startswith("table{"):
            if set(col_dict) == {"dt", "t0", "values"} and len(col_dict) == 3:
                return types.WaveformTable(
                    t0=col_dict["t0"],
                    dt=col_dict["dt"],
                    values=col_dict["values"],
                    attrs=attrs,
                )
            return types.Table(col_dict=col_dict, attrs=attrs)

        if datatype.startswith("struct{"):
            return types.Struct(obj_dict=col_dict, attrs=attrs)

        return types.Struct(obj_dict=col_dict, attrs=attrs or None)

    def _open_source_file(self, filename: str) -> h5py.File:
        """Open one source file with this reader's HDF5 cache settings."""
        return _open_h5(
            filename,
            "r",
            locking=self.locking,
            rdcc_nbytes=self.rdcc_nbytes,
            rdcc_nslots=self.rdcc_nslots,
            rdcc_w0=self.rdcc_w0,
        )

    def _get_source(self, file_id: int) -> _OpenSource:
        pid = os.getpid()
        if self._cache_pid != pid:
            self.close()
            self._cache_pid = pid

        try:
            entry = self._open_sources.pop(file_id)
        except KeyError:
            entry = _OpenSource(h5=self._open_source_file(self.files[file_id]))
        self._open_sources[file_id] = entry

        if self.max_open_files is not None:
            while len(self._open_sources) > self.max_open_files:
                _, old = self._open_sources.popitem(last=False)
                old.h5.close()

        return entry

    def _get_dataset(self, file_id: int, path: str) -> h5py.Dataset:
        return self._get_dataset_from_source(self._get_source(file_id), path)

    @staticmethod
    def _get_dataset_from_source(source: _OpenSource, path: str) -> h5py.Dataset:
        try:
            return source.datasets[path]
        except KeyError:
            ds = source.h5[path]
            source.datasets[path] = ds
            return ds



def _mask_field(node: _SchemaNode, name: str) -> _MaskField:
    """Compile one predicate field for spawned mask workers."""
    datatype = str(node.attrs.get("datatype", ""))
    try:
        is_bool = lh5_datatype.get_nested_datatype_string(datatype) == "bool"
    except Exception:
        is_bool = False
    return _MaskField(name=name, path=node.path, is_bool=is_bool)


def _partition_file_groups(row_counts: np.ndarray, workers: int) -> tuple[tuple[int, ...], ...]:
    """Partition contiguous source files by cumulative row count."""
    n_files = int(row_counts.shape[0])
    n_groups = min(max(1, workers), n_files)
    if n_groups == 1:
        return (tuple(range(n_files)),)

    cumulative = np.cumsum(row_counts, dtype=np.int64)
    total = int(cumulative[-1])
    if total == 0:
        return tuple(
            tuple(int(i) for i in group)
            for group in np.array_split(np.arange(n_files, dtype=np.int64), n_groups)
            if len(group)
        )

    targets = [total * i // n_groups for i in range(1, n_groups)]
    cuts = np.searchsorted(cumulative, targets, side="right")
    cuts = np.unique(np.clip(cuts, 1, n_files - 1))
    bounds = np.concatenate((np.array([0]), cuts, np.array([n_files])))
    return tuple(
        tuple(range(int(a), int(b)))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        if b > a
    )


def _compile_mask_job(job: _MaskJob) -> _MaskResult:
    """Scan one source-file group and return selected physical rows."""
    file_dtype = np.dtype(job.file_dtype)
    local_dtype = np.dtype(job.local_dtype)
    file_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []

    for file_id, filename, n_rows in job.files:
        if n_rows == 0:
            continue

        step = n_rows if job.block_rows is None else int(job.block_rows)
        with _open_h5(
            filename,
            "r",
            locking=job.locking,
            rdcc_nbytes=job.rdcc_nbytes,
            rdcc_nslots=job.rdcc_nslots,
            rdcc_w0=job.rdcc_w0,
        ) as src:
            for start in range(0, n_rows, step):
                stop = min(start + step, n_rows)
                columns: dict[str, np.ndarray] = {}
                for field in job.fields:
                    values = src[field.path][start:stop]
                    if field.is_bool:
                        values = values.astype(np.bool_, copy=False)
                    columns[field.name] = values

                mask = np.ones(stop - start, dtype=np.bool_)
                for name in job.config["quality"]:
                    mask &= np.asarray(columns[name], dtype=np.bool_)
                for name in job.config["not_mask"]:
                    mask &= ~np.asarray(columns[name], dtype=np.bool_)
                for name, intervals in job.config["interval_mask"].items():
                    values = columns[name]
                    accepted = np.zeros(stop - start, dtype=np.bool_)
                    for low, high in intervals.items():
                        accepted |= (values >= low) & (values < high)
                    mask &= accepted

                local = np.flatnonzero(mask)
                if local.size == 0:
                    continue

                local = (local + start).astype(local_dtype, copy=False)
                row_parts.append(local)
                file_parts.append(np.full(local.shape, file_id, dtype=file_dtype))

    if row_parts:
        local_rows = np.concatenate(row_parts)
        file_ids = np.concatenate(file_parts)
    else:
        local_rows = np.empty(0, dtype=local_dtype)
        file_ids = np.empty(0, dtype=file_dtype)

    return _MaskResult(file_ids=file_ids, local_rows=local_rows)

def _task_for_rows(
    *,
    file_id: int,
    rows: np.ndarray,
    output: slice | np.ndarray,
) -> _ReadTask:
    """Build one file task with unique physical I/O rows."""
    rows = np.asarray(rows, dtype=np.int64)
    n_rows = int(rows.shape[0])
    output_ranges = _output_ranges(output)

    if n_rows == 0:
        return _ReadTask(
            file_id=file_id,
            output=output,
            n_rows=0,
            local_rows=rows,
            output_ranges=output_ranges,
        )

    if n_rows == 1:
        return _ReadTask(
            file_id=file_id,
            output=output,
            n_rows=1,
            local_slice=slice(int(rows[0]), int(rows[0]) + 1),
            output_ranges=output_ranges,
        )

    if np.all(rows[1:] > rows[:-1]):
        if np.all(np.diff(rows) == 1):
            return _ReadTask(
                file_id=file_id,
                output=output,
                n_rows=n_rows,
                local_slice=slice(int(rows[0]), int(rows[-1]) + 1),
                output_ranges=output_ranges,
            )
        return _ReadTask(
            file_id=file_id,
            output=output,
            n_rows=n_rows,
            local_rows=rows,
            file_ranges=_rows_to_ranges(rows),
            output_ranges=output_ranges,
        )

    io_rows, gather = np.unique(rows, return_inverse=True)
    gather = gather.astype(np.intp, copy=False)
    if io_rows.size == 1 or np.all(np.diff(io_rows) == 1):
        return _ReadTask(
            file_id=file_id,
            output=output,
            n_rows=n_rows,
            local_slice=slice(int(io_rows[0]), int(io_rows[-1]) + 1),
            gather=gather,
        )

    return _ReadTask(
        file_id=file_id,
        output=output,
        n_rows=n_rows,
        local_rows=io_rows,
        file_ranges=_rows_to_ranges(io_rows),
        gather=gather,
    )


def _read_task_into(
    ds: h5py.Dataset,
    task: _ReadTask,
    out: np.ndarray,
    tail_shape: tuple[int, ...],
) -> None:
    """Read one file task into its final output positions."""
    if task.n_rows == 0:
        return

    if task.gather is None:
        _read_unique_task_into(ds, task, out)
        return

    if task.local_slice is not None:
        n_unique = task.local_slice.stop - task.local_slice.start
        block = np.empty((n_unique, *tail_shape), dtype=ds.dtype)
        ds.read_direct(block, source_sel=np.s_[task.local_slice])
    else:
        rows = task.local_rows
        assert rows is not None
        block = np.empty((rows.shape[0], *tail_shape), dtype=ds.dtype)
        _read_ranges_into(ds, task.file_ranges, block, None)

    out[task.output] = block[task.gather]


def _read_unique_task_into(
    ds: h5py.Dataset,
    task: _ReadTask,
    out: np.ndarray,
) -> None:
    """Read one duplicate-free task directly into the final output buffer."""
    if task.local_slice is not None and isinstance(task.output, slice):
        ds.read_direct(out, source_sel=np.s_[task.local_slice], dest_sel=np.s_[task.output])
        return

    fspace = ds.id.get_space()
    fspace.select_none()
    if task.local_slice is not None:
        _select_row_slice(fspace, task.local_slice)
    else:
        _select_row_ranges(fspace, task.file_ranges)

    mspace = h5py.h5s.create_simple(out.shape)
    mspace.select_none()
    if isinstance(task.output, slice):
        _select_row_slice(mspace, task.output)
    else:
        _select_row_ranges(mspace, task.output_ranges)

    ds.id.read(mspace, fspace, out)


def _read_ranges_into(
    ds: h5py.Dataset,
    ranges: np.ndarray | None,
    out: np.ndarray,
    output_ranges: np.ndarray | None,
) -> None:
    """Read sorted row ranges into a contiguous or selected output buffer."""
    fspace = ds.id.get_space()
    fspace.select_none()
    _select_row_ranges(fspace, ranges)

    mspace = h5py.h5s.create_simple(out.shape)
    if output_ranges is not None:
        mspace.select_none()
        _select_row_ranges(mspace, output_ranges)
    ds.id.read(mspace, fspace, out)


def _rows_to_ranges(rows: np.ndarray) -> np.ndarray:
    """Convert sorted unique row ids into half-open contiguous ranges."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    bounds = np.concatenate((np.array([0]), breaks, np.array([rows.size])))
    ranges = np.empty((len(bounds) - 1, 2), dtype=np.int64)
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        ranges[i, 0] = rows[int(a)]
        ranges[i, 1] = rows[int(b) - 1] + 1
    return ranges


def _output_ranges(output: slice | np.ndarray) -> np.ndarray | None:
    """Return half-open output ranges for an indexed destination."""
    if isinstance(output, slice):
        return None
    return _rows_to_ranges(np.asarray(output, dtype=np.int64))


def _select_row_slice(space: h5py.h5s.SpaceID, rows: slice) -> None:
    """Select one contiguous first-axis slice across all trailing dimensions."""
    start = int(rows.start or 0)
    stop = int(rows.stop or start)
    tail = tuple(int(x) for x in space.shape[1:])
    space.select_hyperslab(
        (start,) + (0,) * len(tail),
        (stop - start, *tail),
        op=h5py.h5s.SELECT_SET,
    )


def _select_row_ranges(space: h5py.h5s.SpaceID, ranges: np.ndarray | None) -> None:
    """Build one HDF5 selection from sorted half-open row ranges."""
    assert ranges is not None
    tail = tuple(int(x) for x in space.shape[1:])
    for i, (start, stop) in enumerate(ranges):
        op = h5py.h5s.SELECT_SET if i == 0 else h5py.h5s.SELECT_OR
        space.select_hyperslab(
            (int(start),) + (0,) * len(tail),
            (int(stop - start), *tail),
            op=op,
        )


def _open_h5(
    filename: str | os.PathLike[str],
    mode: str,
    *,
    locking: bool | None = None,
    rdcc_nbytes: int | None = None,
    rdcc_nslots: int | None = None,
    rdcc_w0: float | None = None,
    **kwargs: Any,
) -> h5py.File:
    """Open an HDF5 file with optional locking and raw chunk-cache settings."""
    if locking is not None:
        kwargs["locking"] = locking
    if rdcc_nbytes is not None:
        kwargs["rdcc_nbytes"] = rdcc_nbytes
    if rdcc_nslots is not None:
        kwargs["rdcc_nslots"] = rdcc_nslots
    if rdcc_w0 is not None:
        kwargs["rdcc_w0"] = rdcc_w0
    return h5py.File(filename, mode, **kwargs)


def _compile_projection(field_mask: Sequence[str] | None) -> Projection | None:
    """Compile flat LH5-style field paths into a nested projection tree."""
    if field_mask is None:
        return None

    root: Projection = {}
    for raw in field_mask:
        parts = [part for part in str(raw).strip("/").split("/") if part]
        if not parts:
            continue

        cursor = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if last:
                cursor[part] = None
                break

            existing = cursor.get(part, "__missing__")
            if existing is None:
                break
            if existing == "__missing__":
                child: Projection = {}
                cursor[part] = child
                cursor = child
            else:
                cursor = existing  # type: ignore[assignment]

    return root


def _mask_fields_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Reconstruct mask-column names from the external configuration."""
    out: list[str] = []
    seen: set[str] = set()

    def add(names: Iterable[str]) -> None:
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)

    add(config["quality"])
    add(config["not_mask"])
    add(config["interval_mask"].keys())
    return tuple(out)


def _selected_leaves(
    node: _SchemaNode,
    projection: Projection | None,
) -> tuple[_SchemaNode, ...]:
    """Return projected dataset leaves in schema order."""
    if node.is_dataset:
        return (node,)

    children = (
        node.children
        if projection is None
        else tuple(child for child in node.children if child.name in projection)
    )
    leaves: list[_SchemaNode] = []
    for child in children:
        child_projection = None if projection is None else projection[child.name]
        leaves.extend(_selected_leaves(child, child_projection))
    return tuple(leaves)


def _declared_group_fields(group: h5py.Group, datatype: str) -> list[str]:
    """Return fields in LH5 datatype order."""
    if datatype.startswith("table{") or datatype.startswith("struct{"):
        return list(lh5_datatype.get_struct_fields(datatype))
    return list(group.keys())


def _first_dataset(node: _SchemaNode) -> _SchemaNode:
    """Return the first projected dataset leaf."""
    if node.is_dataset:
        return node
    for child in node.children:
        try:
            return _first_dataset(child)
        except LookupError:
            pass
    raise LookupError(f"no row-bearing dataset below {node.path}")


def _normalise_attr_value(value: Any) -> Any:
    """Normalize scalar HDF5 attribute values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode()
    return value


def _read_attrs(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Read HDF5 attributes into a plain dictionary."""
    return {str(key): _normalise_attr_value(value) for key, value in obj.attrs.items()}


def _project_group_attrs(
    attrs: Mapping[str, Any],
    selected_fields: Sequence[str],
) -> dict[str, Any]:
    """Update table/struct datatype attributes for a projected field set."""
    out = dict(attrs)
    datatype = str(out.get("datatype", ""))
    if datatype.startswith("table{"):
        out["datatype"] = "table{" + ",".join(selected_fields) + "}"
    elif datatype.startswith("struct{"):
        out["datatype"] = "struct{" + ",".join(selected_fields) + "}"
    return out


def _cast_bool_if_needed(nda: np.ndarray, attrs: Mapping[str, Any]) -> np.ndarray:
    """Cast LH5 bool arrays stored with integer HDF5 dtypes."""
    datatype = str(attrs.get("datatype", ""))
    try:
        nested = lh5_datatype.get_nested_datatype_string(datatype)
    except Exception:
        return nda
    if nested == "bool":
        return nda.astype(np.bool_, copy=False)
    return nda


def _make_lgdo_array(nda: np.ndarray, attrs: Mapping[str, Any]) -> Any:
    """Construct an array-like LGDO from one HDF5 leaf."""
    attrs = dict(attrs)
    datatype = str(attrs["datatype"])
    lgdo_cls = lh5_datatype.datatype(datatype)
    return lgdo_cls(nda=nda, attrs=attrs)


def _logical_indices(idx: Any, length: int) -> np.ndarray:
    """Normalize logical indexing to a one-dimensional int64 array."""
    if idx is None:
        return np.arange(length, dtype=np.int64)

    if isinstance(idx, slice):
        start, stop, step = idx.indices(length)
        return np.arange(start, stop, step, dtype=np.int64)

    arr = np.asarray(idx)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("idx must be scalar, slice, or one-dimensional")

    arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0):
        arr = arr.copy()
        arr[arr < 0] += length
    return arr


def _unsigned_dtype(max_value: int) -> np.dtype:
    """Smallest unsigned integer dtype that can hold ``max_value``."""
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


__all__ = ["VirtualLH5Table"]
