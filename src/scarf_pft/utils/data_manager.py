import os
import glob
import re
import fileinput

import dbetto
from dbetto.catalog import Props
from dbetto import TextDB, AttrsDict

from pathlib import Path

from string import Formatter

class FileDB:
    def __init__(
        self,
        dataflow_dir: str | Path
    ):
        self.dataflow_dir = TextDB(dataflow_dir)

        self.data_dir = self.dataflow_dir.data.__path__
        self.dataflow_cfg = self.dataflow_dir.meta
        self.file_db = self.dataflow_dir["file-db"]

    def _build_pattern(self, mode: str, tier: str, overload=False, **kw):
        """
        Fill the tier-specific file_format template using provided kwargs,
        using '*' wildcards for any missing fields.

        If overload: queries not existing in the pattern will be appended as existing_queries/unknown_queries

        mode:
            parent key in file-db.yaml
        Args:
            **kw: Placeholder values (e.g., datatype, period, run, , timestamp).

        Returns:
            str: A glob-compatible pattern string.
        """
        fmt = self.file_db[mode][tier]
        # collect all placeholders in the format string
        all_keys = {p[1] for p in Formatter().parse(fmt) if p[1]}
        # map provided keys to strings or '*' if None
        sub = {k: (v if v is not None else '*') 
               for k,v in kw.items()}
        # ensure every placeholder has a substitution
        for k in all_keys:
            sub.setdefault(k, '*')

        if overload:
            for k in sub.keys():
                if k not in all_keys:
                    fmt += "/{" + f"{k}" + "}"

        return fmt.format(**sub)

    def _build_regex_from_pattern(self, mode, tier):
        """
        Convert the tier-specific file_format template into a named-group regex
        for parsing metadata fields from actual file paths.

        Returns:
            str: A regex pattern with named groups matching the placeholders.
        """
        pattern = self.file_db[mode][tier]

        if pattern.startswith("/"):
            pattern = pattern[1:]

        seen = set()
        regex = ""
        last = 0
        for m in re.finditer(r"\{(\w+)\}", pattern):
            regex += re.escape(pattern[last:m.start()])
            name = m.group(1)
            if name not in seen:
                regex += fr"(?P<{name}>[^/]+)"
                seen.add(name)
            else:
                regex += fr"(?P={name})"
            last = m.end()
        regex += re.escape(pattern[last:])
        return regex

    def find_data(self, mode, tier, **kwargs):
        """
        Find and return sorted list of files matching the pattern for this tier.

        Args:
            **kwargs: Must include at least experiment, datatype, period, run;
                      timestamp may be omitted to wildcard.

        Returns:
            List[str]: Absolute file paths matching the pattern.
        """
        pattern = self._build_pattern(mode, tier, False, **kwargs).lstrip('/')
        # join and glob
        full_pattern = str(self.data_dir / pattern)
        return sorted(glob.glob(full_pattern))

    def decode_metadata_from_path(self, mode: str, tier: str, path_to_file: str):
        """
        Extract metadata fields (period, datatype, run, etc.) from a file path.

        Args:
            path_to_file (str): File path to parse.

        Raises:
            ValueError: If the path doesn't match the expected format.

        Returns:
            dict: Mapping of placeholder names to their string values.
        """
        norm = path_to_file.replace(os.sep, "/")
        regex = re.compile(self._build_regex_from_pattern(mode, tier))
        m = regex.search(norm)

        if not m:
            raise ValueError(f'Path {path_to_file!r} doesn\'t match pattern {self.file_db[mode][tier]!r}')
        return m.groupdict()

    def parse_metadata_to_path(self, mode: str, target_tier: str, **metadata):
        path = self._build_pattern(
            mode=mode,
            tier=target_tier,
            overload=False,
            **metadata,
        ).lstrip("/")

        return str(self.data_dir / path)

    def get_pars(self, experiment: str, tier: str, detector_type: str, timestamp: str):
        fetching_rule = self.dataflow_cfg[experiment].tier[tier].pars[detector_type].fetching_rule
        result = AttrsDict()
        for rule in fetching_rule:
            Props.add_to(result, self.dataflow_cfg[experiment].tier[rule].pars[detector_type].on(timestamp))

        return result
    
    def get_proc_chain(self, experiment: str, tier: str, detector_type: str, timestamp: str, system: str | None = None):
        # TODO: different dsp config for each channels
        return self.dataflow_cfg[experiment].tier[tier].proc_chain[detector_type].on(timestamp, system=system)

    def lifetime(self, tier: str, files: list[str] | None = None, daq_files: list[str] | None = None):
        if daq_files is None:
            if files is None:
                raise ValueError("Either 'files' or 'daq_files' must be provided.")
            else:
                daq_files = [
                    self.parse_metadata_to_path("file_format", "daq", **self.decode_metadata_from_path("file_format", tier, f))
                    for f in files
                ]

        out = []
        with fileinput.input(daq_files) as f:
            for line in f:
                lifetime_pattern = r'Lifetime:\s*(\d+(?:\.\d+)?)\s*s'
                lifetime_match = re.search(lifetime_pattern, line)
                if lifetime_match is not None:
                    lifetime = float(lifetime_match.group(1))
                    out.append(lifetime)

        if len(out) != len(files):
            raise ValueError(f"Mismatch in number of files and extracted lifetimes: {len(files)} files, {len(out)} lifetimes.")

        return sum(out)
    
    def get_invalid_times(self, experiment: str, timestamp: str):
        return self.dataflow_cfg[experiment].exclude.on(timestamp)

    def _file_is_valid(self, tier: str, file: str):
        metadata = self.decode_metadata_from_path("file_format", tier, file)
        invalid_times = self.get_invalid_times(metadata["experiment"], metadata["timestamp"])

        valid = True
        file_time = dbetto.str_to_datetime(metadata["timestamp"])
        for low, high in invalid_times.items():
            low = dbetto.str_to_datetime(low)
            high = dbetto.str_to_datetime(high)
            if low <= file_time <= high:
                valid = False
                break

        daq_file = self.parse_metadata_to_path("file_format", "daq", **metadata)
        with fileinput.input(daq_file) as f:
            lifetime_exist = False
            for line in f:
                lifetime_pattern = r'Lifetime:\s*(\d+(?:\.\d+)?)\s*s'
                lifetime_match = re.search(lifetime_pattern, line)
                if lifetime_match is not None:
                    lifetime = float(lifetime_match.group(1))
                    lifetime_exist = True
                    break

        if not lifetime_exist:
            valid = False
            print(f'File {os.path.basename(file)} is invalid due to missing lifetime information.')
            print(f'daq line: {line}')
        return valid

    def remove_invalid_files(self, tier, files):
        valid_files = [f for f in files if self._file_is_valid(tier, f)]
        return valid_files

    def get_valid_files(self, mode, tier, **kwargs):
        files = self.find_data(mode, tier, **kwargs)
        valid_files = [f for f in files if self._file_is_valid(tier, f)]
        return valid_files
