#!/usr/bin/env python3

import numpy as np
import json
import time
from privacy.pipeline import PrivacyPipeline
from privacy.audit import AuditWriter, build_audit_entry, build_track_audit_entry
from core.config import load_config


def test_m1_privacy():
    
    cfg = load_config('config/default.yaml')
    print(f'Config loaded: privacy.enabled={cfg.privacy.enabled}')

    pipe = PrivacyPipeline(cfg.privacy)
    print(f'Pipeline created: enabled={pipe._enabled}')

    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    class MockFrame:
        def __init__(self):
            self.image = fake_image
            self.ts = time.perf_counter()
            self.frame_id = 1

    class MockTrack:
        def __init__(self, tid):
            self.track_id = tid
            self.bbox = [100, 100, 200, 200]

    class MockDecision:
        def __init__(self, track_id, person_id='p_0001'):
            self.track_id = track_id
            self.identity_id = person_id
            self.category = 'resident'
            self.confidence = 0.75
            self.binding_state = 'CONFIRMED_STRONG'
            self.id_source = 'F'
            self.extra = {}

    frame = MockFrame()
    tracks = [MockTrack(1), MockTrack(2)]
    decisions = [MockDecision(1), MockDecision(2, person_id=None)]

    pipe.ingest(frame, tracks, decisions)
    print('Ingest successful')

    preview = pipe.get_latest_privacy_frame()
    if preview is not None:
        print(f'Preview frame: shape={preview.shape}')
    else:
        print('Preview frame: None')

    pipe.shutdown()
    print('Shutdown successful')

    audit_path = 'privacy_output/privacy_audit.jsonl'
    with open(audit_path, 'r') as f:
        lines = f.readlines()
        print(f'Audit entries: {len(lines)}')
        for line in lines:
            entry = json.loads(line)
            ts = entry.get('ts')
            track_count = entry.get('track_count')
            print(f'  Entry: ts={ts}, track_count={track_count}')
            if 'tracks' in entry:
                for t in entry['tracks']:
                    tid = t['track_id']
                    pid = t.get('person_id')
                    auth = t.get('authorized')
                    cat = t.get('decision_category')
                    print(f'    Track {tid}: person_id={pid}, authorized={auth}, category={cat}')

    print('\n=== M1 Unit Test PASSED ===')


if __name__ == '__main__':
    test_m1_privacy()
