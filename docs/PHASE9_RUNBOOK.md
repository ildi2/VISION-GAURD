# PHASE 9 OPERATIONAL RUNBOOK
# Chimeric Continuity Mode - Production Rollout Procedures

**Document Version:** 1.0  
**Last Updated:** 2026-01-31  
**Owner:** GaitGuard Operations Team  
**Status:** Production-Ready

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Rollout Stages](#rollout-stages)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Validation](#monitoring--validation)
5. [Rollback Procedures](#rollback-procedures)
6. [Incident Response](#incident-response)
7. [Success Criteria](#success-criteria)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## OVERVIEW

### Purpose

This runbook provides step-by-step procedures for deploying chimeric continuity mode to production through a 4-week staged rollout. The staged approach minimizes risk by validating each component before enabling the next.

### Prerequisites

- ✅ Phase 8 complete (all scenario tests passing)
- ✅ System running stable in classic mode
- ✅ Backup procedures verified
- ✅ Monitoring tools deployed
- ✅ Operations team trained

### Rollout Timeline

| Week | Stage | Mode | Risk | Focus |
|------|-------|------|------|-------|
| 1 | Shadow Mode | shadow_continuity | Zero | Observation, tuning |
| 2 | Strict Guards | continuity | Low | Safety validation |
| 3 | Relaxed Guards | continuity | Medium | Performance optimization |
| 4 | Grace Enabled | continuity | Medium | Full feature set |
| 5+ | Production | continuity | Low | Ongoing monitoring |

---

## ROLLOUT STAGES

### Stage 1: Shadow Mode (Week 1)

**Objective:** Observe continuity behavior without affecting production

**Duration:** 1 week  
**Risk Level:** Zero (observe-only, no mutations)

**Configuration:**
```yaml
chimeric:
  mode: "shadow_continuity"
  continuity:
    shadow_mode: true
    shadow_metrics_log_interval_sec: 10  # More frequent for tuning
```

**Activities:**
1. Deploy shadow mode configuration
2. Monitor shadow metrics logs
3. Collect baseline data:
   - Typical carry rate
   - Guard break distribution
   - Appearance distance patterns
   - BBox movement patterns
4. Tune thresholds if needed
5. Validate no system impact (FPS, memory, stability)

**Success Criteria:**
- [ ] No crashes or exceptions
- [ ] Shadow metrics logged successfully
- [ ] Baseline data collected
- [ ] Thresholds tuned (if needed)
- [ ] Ready for real mode

**Rollback:** Not needed (shadow mode has no production impact)

---

### Stage 2: Strict Guards (Week 2)

**Objective:** Enable real continuity with conservative thresholds

**Duration:** 1 week  
**Risk Level:** Low (strict guards minimize false carries)

**Configuration:**
```yaml
chimeric:
  mode: "continuity"
  continuity:
    shadow_mode: false
    min_track_age_frames: 15           # Stricter than production
    appearance_distance_threshold: 0.30 # Stricter than production
    max_lost_frames: 1                  # Stricter than production
    grace_window_seconds: 0.0           # Grace disabled
```

**Activities:**
1. Deploy strict guards configuration
2. Restart GaitGuard pipeline
3. Monitor real-time metrics:
   - Identity persistence rate
   - False carry incidents (critical safety check)
   - System performance (FPS impact)
4. Collect operator feedback
5. Validate success criteria

**Success Criteria:**
- [ ] False carry rate <1% (CRITICAL)
- [ ] Zero system crashes
- [ ] FPS degradation <5%
- [ ] Operator feedback positive
- [ ] System stable for 7 days

**Rollback Triggers:**
- False carry rate >1%
- System crashes or instability
- FPS degradation >10%

---

### Stage 3: Relaxed Guards (Week 3)

**Objective:** Achieve production thresholds for optimal persistence

**Duration:** 1 week  
**Risk Level:** Medium (increased carry rate, monitor closely)

**Configuration:**
```yaml
chimeric:
  mode: "continuity"
  continuity:
    shadow_mode: false
    min_track_age_frames: 10            # Production target
    appearance_distance_threshold: 0.35 # Production target
    max_lost_frames: 5                  # Production target
    grace_window_seconds: 0.0           # Grace still disabled
```

**Activities:**
1. Deploy relaxed guards configuration
2. Restart GaitGuard pipeline
3. Monitor metrics closely:
   - Identity persistence (target ≥95%)
   - False carry rate (must stay <1%)
   - Carry rate increase vs Stage 2
4. Compare with Stage 2 metrics
5. Validate success criteria

**Success Criteria:**
- [ ] Identity persistence ≥95% (PRIMARY GOAL)
- [ ] False carry rate <1% (SAFETY)
- [ ] Zero system crashes
- [ ] FPS degradation <5%
- [ ] Operator feedback positive

**Rollback Triggers:**
- False carry rate >1%
- Identity persistence <90%
- System instability

---

### Stage 4: Grace Reattachment (Week 4)

**Objective:** Enable grace reattachment for brief signal loss recovery

**Duration:** 1 week  
**Risk Level:** Medium (new feature, requires validation)

**Configuration:**
```yaml
chimeric:
  mode: "continuity"
  continuity:
    shadow_mode: false
    # All production thresholds from Stage 3
    grace_window_seconds: 1.0  # ENABLED (1 second window)
```

**Activities:**
1. Deploy grace-enabled configuration
2. Restart GaitGuard pipeline
3. Monitor grace metrics:
   - Reattachment success rate (target ≥80%)
   - False reattachment rate (must be <1%)
   - Grace pool size (should not grow unbounded)
4. Validate success criteria

**Success Criteria:**
- [ ] Grace reattachment success ≥80%
- [ ] False reattachment rate <1%
- [ ] Identity persistence maintained (≥95%)
- [ ] Zero system crashes
- [ ] Grace pool size stable

**Rollback Triggers:**
- False reattachment rate >1%
- Grace pool growing unbounded
- System instability

---

### Stage 5: Full Production (Ongoing)

**Objective:** Maintain production deployment with continuous monitoring

**Duration:** Ongoing (permanent production state)  
**Risk Level:** Low (validated through Stages 1-4)

**Configuration:**
```yaml
chimeric:
  mode: "continuity"
  continuity:
    shadow_mode: false
    shadow_metrics_log_interval_sec: 60  # Less frequent (every minute)
    # All production-tuned thresholds
```

**Activities:**
1. Deploy final production configuration
2. Continuous monitoring
3. Monthly metric review
4. Threshold tuning as needed
5. Incident response (if needed)

**Ongoing Monitoring:**
- Identity persistence ≥95%
- False carry rate <1%
- Grace reattachment ≥80%
- System performance <5% FPS impact

**Maintenance:**
- Monthly metrics review
- Quarterly threshold tuning
- Annual configuration audit

---

## DEPLOYMENT PROCEDURES

### Using Rollout Manager (Automated)

**Deploy Stage N:**
```bash
# Navigate to project root
cd /path/to/GaitGuard

# Activate virtual environment
source .venv310/bin/activate  # Linux/Mac
.venv310\Scripts\activate     # Windows

# Deploy stage (with confirmation)
python scripts/rollout/rollout_manager.py deploy --stage N

# Deploy stage (skip confirmation, for automation)
python scripts/rollout/rollout_manager.py deploy --stage N --force

# Check current status
python scripts/rollout/rollout_manager.py status
```

**What Rollout Manager Does:**
1. Validates stage number (1-5)
2. Displays stage information (duration, risk, description)
3. Prompts for confirmation (unless --force)
4. Loads base config and stage config
5. Merges configurations
6. Validates merged config
7. Creates backup of current config
8. Deploys new configuration
9. Displays next steps

**Restart GaitGuard After Deployment:**
```bash
# Stop current process (if running)
pkill -f "python.*main_loop.py"  # Linux/Mac
# OR: Find and stop process manually on Windows

# Start GaitGuard
python -m core.main_loop
```

---

### Manual Deployment (Fallback)

If rollout manager unavailable, deploy manually:

**Step 1: Create Backup**
```bash
# Create backup directory
mkdir -p config/backups

# Backup current config
cp config/default.yaml config/backups/default_$(date +%Y%m%d_%H%M%S).yaml
```

**Step 2: Merge Configuration**
```bash
# Copy stage config to temporary file
cp scripts/rollout/stageN_*.yaml /tmp/stage_config.yaml

# Manually merge chimeric section into config/default.yaml
# Edit config/default.yaml, replace chimeric section with stage config
```

**Step 3: Validate**
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/default.yaml'))"

# If no errors, config is valid
```

**Step 4: Restart GaitGuard**
```bash
# Stop and restart as above
```

---

## MONITORING & VALIDATION

### Real-Time Monitoring

**Using Metrics Collector:**
```bash
# Real-time monitoring (updates every 10 seconds)
python scripts/rollout/metrics_collector.py monitor \
    --logfile logs/gaitguard.log \
    --interval 10
```

**What You'll See:**
```
==============================================================
CONTINUITY METRICS SUMMARY
==============================================================
Time Period: 2026-01-31 14:00:00 to 2026-01-31 14:30:00
Duration: 0:30:00

Frames Analyzed: 54000
Total Tracks: 1250

--- ID Source Distribution ---
Face-Assigned (F):    450 ( 36.0%)
GPS-Carried   (G):    700 ( 56.0%)
Unknown       (U):    100 (  8.0%)

--- Key Metrics ---
GPS Carry Rate:        56.00%
Identity Persistence:  92.00%

--- Grace Reattachment ---
Attempts:          45
Successes:         38 ( 84.4%)
Failures:           7

--- Guard Breaks ---
appearance_break    :     12 ( 48.0%)
bbox_break          :      8 ( 32.0%)
health_break        :      5 ( 20.0%)

--- Performance ---
Average FPS:  28.50
Min FPS:      26.00
Max FPS:      30.00
==============================================================
```

---

### Historical Analysis

**Generate Report:**
```bash
# Analyze last 1 hour
python scripts/rollout/metrics_collector.py analyze \
    --logfile logs/gaitguard.log \
    --duration 1h

# Analyze last 24 hours
python scripts/rollout/metrics_collector.py analyze \
    --logfile logs/gaitguard.log \
    --duration 1d

# Generate report file
python scripts/rollout/metrics_collector.py report \
    --logfile logs/gaitguard.log \
    --duration 1h \
    --output report_$(date +%Y%m%d).txt
```

---

### Log Inspection

**Key Log Messages to Monitor:**

**Shadow Metrics (Stage 1):**
```
Shadow metrics | carries=45 binds=10 app_breaks=2 bbox_breaks=1 health_breaks=0 grace=3
```

**Guard Breaks:**
```
APPEARANCE BREAK: track_id=3 | distance=0.42 > threshold=0.35
BBOX TELEPORT: track_id=5 | center_dist=1565.2px (thresh=550.7) | IoU=0.000
HEALTH BREAK: track_id=7 | lost_frames=3 > threshold=2
```

**Grace Reattachment:**
```
Grace reattachment: track_id=5 → 7 (success) | distance=0.12 bbox_dist=85px
Grace reattachment: track_id=8 → 12 (failure) | no candidates within grace window
```

**Errors to Watch:**
```
ERROR - Continuity binder exception: [exception details]
ERROR - Memory leak detected: grace pool size=1000 (growing)
ERROR - False carry detected: track switched from alice to bob
```

---

## ROLLBACK PROCEDURES

### Emergency Rollback (Immediate - 30 seconds)

**When to Use:**
- Critical production incident
- False carry rate spiking
- System crashes
- Unacceptable performance degradation

**Procedure:**
```bash
# Option 1: Using rollout manager (recommended)
python scripts/rollout/rollout_manager.py emergency-disable

# Option 2: Manual (if rollout manager unavailable)
# Edit config/default.yaml:
chimeric:
  mode: "classic"  # Change from "continuity" to "classic"

# Restart GaitGuard
pkill -f "python.*main_loop.py"
python -m core.main_loop
```

**Result:** System immediately reverts to classic mode (continuity disabled)

---

### Staged Rollback (Gradual)

**When to Use:**
- Metrics not meeting targets
- Issues with specific features
- Need to investigate problems

**Procedure:**
```bash
# Rollback to previous stage
python scripts/rollout/rollout_manager.py rollback

# OR: Rollback to specific backup
python scripts/rollout/rollout_manager.py rollback \
    --backup config/backups/default_20260131_140000.yaml
```

**Rollback Options:**
1. **Stage 4 → Stage 3:** Disable grace reattachment only
2. **Stage 3 → Stage 2:** Tighten guards to strict thresholds
3. **Stage 2 → Stage 1:** Switch to shadow mode (observe-only)
4. **Any Stage → Classic:** Emergency disable (full rollback)

---

### Partial Rollback (Feature-Specific)

**Disable Grace Reattachment Only:**
```yaml
chimeric:
  mode: "continuity"
  continuity:
    grace_window_seconds: 0.0  # Disable grace, keep core carry
```

**Tighten Specific Guard:**
```yaml
continuity:
  appearance_distance_threshold: 0.25  # Much stricter (from 0.35)
```

---

## INCIDENT RESPONSE

### Critical Incidents

**Definition:** Production-impacting issues requiring immediate action

**Examples:**
- False carry rate >1%
- System crashes
- FPS degradation >20%
- Memory leak

**Response Procedure:**

**Step 1: Immediate Action (5 minutes)**
```bash
# Execute emergency rollback
python scripts/rollout/rollout_manager.py emergency-disable

# Restart system
pkill -f "python.*main_loop.py"
python -m core.main_loop

# Verify classic mode active
tail -f logs/gaitguard.log | grep "chimeric.*mode"
# Should see: "chimeric mode: classic"
```

**Step 2: Incident Assessment (30 minutes)**
1. Collect logs: `cp logs/gaitguard.log logs/incident_$(date +%Y%m%d_%H%M%S).log`
2. Generate metrics report: `python scripts/rollout/metrics_collector.py report ...`
3. Document symptoms, timeline, impact
4. Identify root cause (logs, metrics, code review)

**Step 3: Root Cause Analysis (1-2 hours)**
1. Review incident logs
2. Reproduce issue in test environment (if possible)
3. Identify fix (code change, threshold tuning, procedure update)
4. Document findings

**Step 4: Resolution (varies)**
1. Implement fix
2. Validate in test environment
3. Update runbook/procedures
4. Plan re-deployment (if appropriate)

---

### Non-Critical Issues

**Definition:** Issues that don't require immediate rollback

**Examples:**
- Metrics slightly below targets (persistence 93% vs 95% target)
- Occasional guard breaks
- Minor FPS variation

**Response Procedure:**
1. Document issue
2. Monitor trend (is it worsening?)
3. Analyze root cause
4. Plan threshold tuning
5. Deploy tuning in next maintenance window

---

## SUCCESS CRITERIA

### Stage-Specific Criteria

| Stage | Primary Metric | Target | Rollback Trigger |
|-------|----------------|--------|------------------|
| 1 | System Stability | Zero crashes | Any crash |
| 2 | False Carry Rate | <1% | >1% |
| 3 | Identity Persistence | ≥95% | <90% |
| 4 | Grace Success | ≥80% | <70% or false reattach >1% |
| 5 | All Metrics | Maintained | Any metric miss |

### Production Metrics (Ongoing)

**Quantitative:**
- Identity Persistence: ≥95%
- False Carry Rate: <1%
- Grace Reattachment Success: ≥80%
- FPS Impact: <5% degradation
- System Uptime: ≥99.9%

**Qualitative:**
- Operator feedback positive
- Visual coherence maintained
- Operational confidence high

---

## TROUBLESHOOTING GUIDE

### Issue: False Carries Detected

**Symptoms:** Identity carried to wrong person

**Diagnosis:**
```bash
# Review guard break logs
grep "APPEARANCE BREAK" logs/gaitguard.log
grep "tracker switched" logs/gaitguard.log
```

**Solutions:**
1. Tighten appearance threshold:
   ```yaml
   appearance_distance_threshold: 0.30  # From 0.35
   ```
2. Increase track age requirement:
   ```yaml
   min_track_age_frames: 15  # From 10
   ```
3. Review tracker performance (OC-SORT issues?)

---

### Issue: Identity Persistence Too Low

**Symptoms:** Persistence <95%, frequent guard breaks

**Diagnosis:**
```bash
# Analyze guard break distribution
python scripts/rollout/metrics_collector.py analyze \
    --logfile logs/gaitguard.log --duration 1h
```

**Solutions:**
1. If appearance breaks dominant → loosen threshold:
   ```yaml
   appearance_distance_threshold: 0.40  # From 0.35
   ```
2. If bbox breaks dominant → increase displacement:
   ```yaml
   max_bbox_displacement_px: 800  # From 600
   ```
3. If health breaks dominant → relax health criteria:
   ```yaml
   max_lost_frames: 7  # From 5
   ```

---

### Issue: Grace Pool Growing Unbounded

**Symptoms:** Memory usage increasing, grace pool size >1000

**Diagnosis:**
```bash
# Check grace pool size in logs
grep "grace pool size" logs/gaitguard.log
```

**Solutions:**
1. Reduce grace window:
   ```yaml
   grace_window_seconds: 0.5  # From 1.0
   ```
2. Investigate grace expiry (is cleanup working?)
3. Check for memory leak (code bug?)

---

### Issue: FPS Degradation

**Symptoms:** FPS dropped >5% vs baseline

**Diagnosis:**
```bash
# Compare FPS before/after continuity
grep "FPS" logs/gaitguard_classic.log | tail -100
grep "FPS" logs/gaitguard_continuity.log | tail -100
```

**Solutions:**
1. Profile performance (which guard is slow?)
2. Reduce appearance EMA computation frequency
3. Optimize bbox IoU calculation (if needed)
4. Consider hardware upgrade (if persistent)

---

### Issue: System Crashes

**Symptoms:** GaitGuard process terminates unexpectedly

**Diagnosis:**
```bash
# Check for exceptions in logs
grep "ERROR\|EXCEPTION\|Traceback" logs/gaitguard.log | tail -100

# Check system logs
dmesg | grep -i "killed\|oom"  # Linux
# Check Event Viewer on Windows
```

**Solutions:**
1. Emergency rollback to classic mode
2. Analyze exception traceback
3. Fix code bug (if identified)
4. Check for memory leak
5. Validate test coverage (add regression test)

---

## APPENDICES

### A. Configuration Reference

See `scripts/rollout/stage*.yaml` files for complete stage configurations.

### B. Metrics Reference

See `scripts/rollout/metrics_collector.py` for metric definitions.

### C. Emergency Contacts

- **Operations Lead:** [Name] - [Contact]
- **Engineering Lead:** [Name] - [Contact]
- **On-Call Engineer:** [Rotation] - [Contact]

### D. Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-01-31 | 1.0 | Initial runbook | GaitGuard Team |

---

**END OF RUNBOOK**

This runbook provides comprehensive operational procedures for Phase 9 production rollout. Follow staged approach, monitor metrics closely, and rollback immediately if issues arise.
