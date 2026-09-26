import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, List

class StateManager:
    """
    Restartable pipeline state and integrity manager.
    Atomically checkpoints progress to progress.json and manifest.json.
    """
    def __init__(self, progress_path: Path, manifest_path: Path):
        self.progress_path = Path(progress_path)
        self.manifest_path = Path(manifest_path)
        self._ensure_init()
        
    def _ensure_init(self):
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.progress_path.exists():
            initial_progress = {
                'stage_1_data_loading': {'status': 'PENDING'},
                'stage_2_normalization': {'status': 'PENDING'},
                'stage_3_blocking': {'status': 'PENDING'},
                'stage_4_feature_engineering': {'status': 'PENDING'},
                'stage_5_training': {'status': 'PENDING'},
                'stage_6_inference': {'status': 'PENDING'}
            }
            self._atomic_write(self.progress_path, initial_progress)
            
        if not self.manifest_path.exists():
            initial_manifest = {
                'created_at': time.time(),
                'sources': {},
                'artifacts': {}
            }
            self._atomic_write(self.manifest_path, initial_manifest)

    def _atomic_write(self, path: Path, data: Dict[str, Any]):
        tmp_path = path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load_progress(self) -> Dict[str, Any]:
        with open(self.progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def is_stage_completed(self, stage_name: str) -> bool:
        prog = self.load_progress()
        return prog.get(stage_name, {}).get('status') == 'COMPLETED'

    def mark_in_progress(self, stage_name: str, meta: Optional[Dict[str, Any]] = None):
        prog = self.load_progress()
        prog[stage_name] = {
            'status': 'IN_PROGRESS',
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'metadata': meta or {}
        }
        self._atomic_write(self.progress_path, prog)

    def mark_completed(self, stage_name: str, meta: Optional[Dict[str, Any]] = None):
        prog = self.load_progress()
        prog[stage_name] = {
            'status': 'COMPLETED',
            'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'metadata': meta or {}
        }
        self._atomic_write(self.progress_path, prog)

    # --- Source Fingerprinting & Chunk Registry (Adjustment 2) ---

    @staticmethod
    def compute_file_fingerprint(path: Path) -> Dict[str, Any]:
        p = Path(path)
        stat = p.stat()
        hasher = hashlib.md5()
        with open(p, 'rb') as f:
            chunk = f.read(65536)
            hasher.update(chunk)
        return {
            'path': str(p.resolve()),
            'size_bytes': stat.st_size,
            'mtime': stat.st_mtime,
            'md5_head_64k': hasher.hexdigest()
        }

    def register_source_file(self, source_key: str, path: Path) -> Dict[str, Any]:
        manifest = self.load_manifest()
        if 'sources' not in manifest:
            manifest['sources'] = {}
            
        current_fp = self.compute_file_fingerprint(path)
        existing = manifest['sources'].get(source_key)
        
        # Check if file has changed
        if existing and (
            existing.get('size_bytes') != current_fp['size_bytes'] or
            existing.get('md5_head_64k') != current_fp['md5_head_64k']
        ):
            manifest['sources'][source_key] = {
                **current_fp,
                'registered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'total_rows_processed': 0,
                'completed_chunks': []
            }
        elif not existing:
            manifest['sources'][source_key] = {
                **current_fp,
                'registered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'total_rows_processed': 0,
                'completed_chunks': []
            }
            
        self._atomic_write(self.manifest_path, manifest)
        return manifest['sources'][source_key]

    def is_chunk_completed(self, source_key: str, chunk_idx: int) -> bool:
        manifest = self.load_manifest()
        source_info = manifest.get('sources', {}).get(source_key, {})
        return chunk_idx in source_info.get('completed_chunks', [])

    def mark_chunk_completed(self, source_key: str, chunk_idx: int, rows_in_chunk: int, partition_files: Optional[Dict[str, str]] = None):
        manifest = self.load_manifest()
        if 'sources' not in manifest:
            manifest['sources'] = {}
        if source_key not in manifest['sources']:
            manifest['sources'][source_key] = {'completed_chunks': [], 'total_rows_processed': 0}
            
        src = manifest['sources'][source_key]
        if 'completed_chunks' not in src:
            src['completed_chunks'] = []
            
        if chunk_idx not in src['completed_chunks']:
            src['completed_chunks'].append(chunk_idx)
            src['completed_chunks'].sort()
            src['total_rows_processed'] = src.get('total_rows_processed', 0) + rows_in_chunk
            
        if partition_files:
            if 'partitions' not in src:
                src['partitions'] = {}
            for country, p_path in partition_files.items():
                if country not in src['partitions']:
                    src['partitions'][country] = []
                if p_path not in src['partitions'][country]:
                    src['partitions'][country].append(p_path)
                    
        self._atomic_write(self.manifest_path, manifest)

    def record_artifact(self, name: str, file_path: Path, row_count: int, meta: Optional[Dict[str, Any]] = None):
        manifest = self.load_manifest()
        if 'artifacts' not in manifest:
            manifest['artifacts'] = {}
        p = Path(file_path)
        manifest['artifacts'][name] = {
            'path': str(p.resolve()),
            'size_bytes': p.stat().st_size if p.exists() else 0,
            'row_count': row_count,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'metadata': meta or {}
        }
        self._atomic_write(self.manifest_path, manifest)
