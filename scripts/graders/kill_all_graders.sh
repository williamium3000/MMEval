#!/bin/bash
# Script to kill all running grader processes (mmhal, ha_dpo, chair)

echo "=================================================================================="
echo "Killing all grader processes..."
echo "=================================================================================="

# Function to kill processes by pattern
kill_by_pattern() {
    local pattern="$1"
    local name="$2"
    
    # Find PIDs matching the pattern
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -z "$pids" ]; then
        echo "No $name processes found"
        return 0
    fi
    
    echo "Found $name processes: $pids"
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait a bit and force kill if still running
    sleep 2
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Force killing PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    echo "✓ $name processes killed"
    echo ""
}

# Kill mmhal processes
kill_by_pattern "grader/mmhal/mmhal_grader.py" "mmhal"

# Kill ha_dpo processes
kill_by_pattern "grader/ha_dpo_grader/eval.py" "ha_dpo"

# Kill chair processes
kill_by_pattern "grader/chair/chair_dyna_vg.py" "chair"

echo "=================================================================================="
echo "All grader processes killed!"
echo "=================================================================================="

# Show remaining processes (if any)
echo ""
echo "Checking for remaining processes..."
remaining=$(pgrep -f "grader/(mmhal|ha_dpo_grader|chair)" 2>/dev/null)
if [ -n "$remaining" ]; then
    echo "Warning: Some processes may still be running: $remaining"
    echo "You may need to kill them manually: kill -9 $remaining"
else
    echo "✓ No remaining grader processes found"
fi
