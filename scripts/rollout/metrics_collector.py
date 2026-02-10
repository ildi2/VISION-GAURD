#!/usr/bin/env python3

import argparse
import logging
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class ContinuityMetrics:
    
    def __init__(self):
        self.total_frames = 0
        self.total_tracks = 0
        
        self.face_assigned = 0
        self.gps_carried = 0
        self.unknown = 0
        
        self.guard_breaks = Counter()
        
        self.grace_attempts = 0
        self.grace_successes = 0
        self.grace_failures = 0
        
        self.fps_samples = []
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def add_id_source(self, source: str):
        if source == "F":
            self.face_assigned += 1
        elif source == "G":
            self.gps_carried += 1
        elif source == "U":
            self.unknown += 1
    
    def add_guard_break(self, reason: str):
        self.guard_breaks[reason] += 1
    
    def add_grace_event(self, success: bool):
        self.grace_attempts += 1
        if success:
            self.grace_successes += 1
        else:
            self.grace_failures += 1
    
    def add_fps_sample(self, fps: float):
        self.fps_samples.append(fps)
    
    @property
    def carry_rate(self) -> float:
        total_id_decisions = self.face_assigned + self.gps_carried + self.unknown
        if total_id_decisions == 0:
            return 0.0
        return (self.gps_carried / total_id_decisions) * 100.0
    
    @property
    def persistence_rate(self) -> float:
        total_id_decisions = self.face_assigned + self.gps_carried + self.unknown
        if total_id_decisions == 0:
            return 0.0
        assigned = self.face_assigned + self.gps_carried
        return (assigned / total_id_decisions) * 100.0
    
    @property
    def grace_success_rate(self) -> float:
        if self.grace_attempts == 0:
            return 0.0
        return (self.grace_successes / self.grace_attempts) * 100.0
    
    @property
    def avg_fps(self) -> float:
        if not self.fps_samples:
            return 0.0
        return sum(self.fps_samples) / len(self.fps_samples)
    
    def summary(self) -> str:
        lines = []
        lines.append("="*60)
        lines.append("CONTINUITY METRICS SUMMARY")
        lines.append("="*60)
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            lines.append(f"Time Period: {self.start_time} to {self.end_time}")
            lines.append(f"Duration: {duration}")
        
        lines.append(f"\nFrames Analyzed: {self.total_frames}")
        lines.append(f"Total Tracks: {self.total_tracks}")
        
        lines.append(f"\n--- ID Source Distribution ---")
        total = self.face_assigned + self.gps_carried + self.unknown
        if total > 0:
            lines.append(f"Face-Assigned (F): {self.face_assigned:6d} ({self.face_assigned/total*100:5.1f}%)")
            lines.append(f"GPS-Carried   (G): {self.gps_carried:6d} ({self.gps_carried/total*100:5.1f}%)")
            lines.append(f"Unknown       (U): {self.unknown:6d} ({self.unknown/total*100:5.1f}%)")
        
        lines.append(f"\n--- Key Metrics ---")
        lines.append(f"GPS Carry Rate:       {self.carry_rate:6.2f}%")
        lines.append(f"Identity Persistence: {self.persistence_rate:6.2f}%")
        
        if self.grace_attempts > 0:
            lines.append(f"\n--- Grace Reattachment ---")
            lines.append(f"Attempts:     {self.grace_attempts:6d}")
            lines.append(f"Successes:    {self.grace_successes:6d} ({self.grace_success_rate:5.1f}%)")
            lines.append(f"Failures:     {self.grace_failures:6d}")
        
        if self.guard_breaks:
            lines.append(f"\n--- Guard Breaks ---")
            total_breaks = sum(self.guard_breaks.values())
            for reason, count in self.guard_breaks.most_common():
                pct = (count / total_breaks * 100) if total_breaks > 0 else 0
                lines.append(f"{reason:20s}: {count:6d} ({pct:5.1f}%)")
        
        if self.fps_samples:
            lines.append(f"\n--- Performance ---")
            lines.append(f"Average FPS: {self.avg_fps:6.2f}")
            lines.append(f"Min FPS:     {min(self.fps_samples):6.2f}")
            lines.append(f"Max FPS:     {max(self.fps_samples):6.2f}")
        
        lines.append("="*60)
        return "\n".join(lines)


def parse_log_line(line: str, metrics: ContinuityMetrics):
    
    shadow_pattern = r'Shadow metrics.*carries=(\d+).*binds=(\d+).*app_breaks=(\d+).*bbox_breaks=(\d+).*health_breaks=(\d+).*grace=(\d+)'
    match = re.search(shadow_pattern, line)
    if match:
        carries = int(match.group(1))
        binds = int(match.group(2))
        app_breaks = int(match.group(3))
        bbox_breaks = int(match.group(4))
        health_breaks = int(match.group(5))
        grace = int(match.group(6))
        
        metrics.gps_carried = carries
        metrics.face_assigned = binds
        metrics.guard_breaks['appearance_break'] = app_breaks
        metrics.guard_breaks['bbox_break'] = bbox_breaks
        metrics.guard_breaks['health_break'] = health_breaks
        metrics.grace_successes = grace
        return
    
    id_source_pattern = r'id_source=([FGU])'
    match = re.search(id_source_pattern, line)
    if match:
        source = match.group(1)
        metrics.add_id_source(source)
        metrics.total_tracks += 1
        return
    
    if 'APPEARANCE BREAK' in line:
        metrics.add_guard_break('appearance_break')
        return
    
    if 'BBOX TELEPORT' in line:
        metrics.add_guard_break('bbox_break')
        return
    
    if 'HEALTH BREAK' in line:
        metrics.add_guard_break('health_break')
        return
    
    if 'CONTRADICTION BREAK' in line:
        metrics.add_guard_break('contradiction_break')
        return
    
    if 'young_track_skips' in line or 'Too young' in line:
        metrics.add_guard_break('young_track')
        return
    
    grace_pattern = r'Grace reattachment.*\((success|failure)\)'
    match = re.search(grace_pattern, line)
    if match:
        success = match.group(1) == 'success'
        metrics.add_grace_event(success)
        return
    
    fps_pattern = r'FPS[:\s]+(\d+\.?\d*)'
    match = re.search(fps_pattern, line)
    if match:
        fps = float(match.group(1))
        metrics.add_fps_sample(fps)
        return
    
    if 'Processing frame' in line or 'Frame ' in line:
        metrics.total_frames += 1
        return


def analyze_logfile(logfile: Path, duration_hours: Optional[float] = None) -> ContinuityMetrics:
    if not logfile.exists():
        log.error(f"Log file not found: {logfile}")
        return ContinuityMetrics()
    
    metrics = ContinuityMetrics()
    cutoff_time = None
    
    if duration_hours:
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        log.info(f"Analyzing logs since {cutoff_time}")
    
    log.info(f"Reading log file: {logfile}")
    
    with open(logfile, 'r') as f:
        for line in f:
            timestamp_pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
            match = re.match(timestamp_pattern, line)
            
            if match and cutoff_time:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    if timestamp < cutoff_time:
                        continue
                    
                    if not metrics.start_time:
                        metrics.start_time = timestamp
                    metrics.end_time = timestamp
                except:
                    pass
            
            parse_log_line(line, metrics)
    
    log.info(f"Analysis complete: {metrics.total_frames} frames, {metrics.total_tracks} tracks")
    return metrics


def monitor_realtime(logfile: Path, interval_sec: int = 10):
    import time
    
    log.info(f"Monitoring {logfile} (Ctrl+C to stop)")
    log.info(f"Update interval: {interval_sec} seconds\n")
    
    try:
        with open(logfile, 'r') as f:
            f.seek(0, 2)
            
            metrics = ContinuityMetrics()
            last_update = time.time()
            
            while True:
                line = f.readline()
                
                if line:
                    parse_log_line(line, metrics)
                else:
                    now = time.time()
                    if now - last_update >= interval_sec:
                        print("\033[H\033[J")
                        print(metrics.summary())
                        last_update = now
                    
                    time.sleep(0.1)
    
    except KeyboardInterrupt:
        log.info("\nMonitoring stopped")
        print(metrics.summary())
    except Exception as e:
        log.error(f"Monitoring error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Continuity Metrics Collector - Phase 9 Monitoring"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--logfile', type=str, required=True,
                               help='Path to log file')
    monitor_parser.add_argument('--interval', type=int, default=10,
                               help='Update interval (seconds)')
    
    analyze_parser = subparsers.add_parser('analyze', help='Historical analysis')
    analyze_parser.add_argument('--logfile', type=str, required=True,
                               help='Path to log file')
    analyze_parser.add_argument('--duration', type=str,
                               help='Time window (e.g., "1h", "30m", "2d")')
    
    report_parser = subparsers.add_parser('report', help='Generate report')
    report_parser.add_argument('--logfile', type=str, required=True,
                              help='Path to log file')
    report_parser.add_argument('--output', type=str,
                              help='Output file (default: stdout)')
    report_parser.add_argument('--duration', type=str,
                              help='Time window (e.g., "1h", "30m", "2d")')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'monitor':
        logfile = Path(args.logfile)
        monitor_realtime(logfile, args.interval)
    
    elif args.command == 'analyze':
        logfile = Path(args.logfile)
        
        duration_hours = None
        if args.duration:
            match = re.match(r'(\d+)([hmd])', args.duration)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if unit == 'h':
                    duration_hours = value
                elif unit == 'm':
                    duration_hours = value / 60.0
                elif unit == 'd':
                    duration_hours = value * 24.0
        
        metrics = analyze_logfile(logfile, duration_hours)
        print(metrics.summary())
    
    elif args.command == 'report':
        logfile = Path(args.logfile)
        
        duration_hours = None
        if args.duration:
            match = re.match(r'(\d+)([hmd])', args.duration)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if unit == 'h':
                    duration_hours = value
                elif unit == 'm':
                    duration_hours = value / 60.0
                elif unit == 'd':
                    duration_hours = value * 24.0
        
        metrics = analyze_logfile(logfile, duration_hours)
        report = metrics.summary()
        
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(report)
            log.info(f"Report written to: {output_path}")
        else:
            print(report)


if __name__ == '__main__':
    main()
