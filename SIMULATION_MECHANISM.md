# PIM Simulator - Event-Driven Execution Mechanism

## Table of Contents
- [1. Overview](#1-overview)
- [2. System Architecture](#2-system-architecture)
- [3. Core Concepts](#3-core-concepts)
- [4. Execution Flow](#4-execution-flow)
- [5. Example Scenario](#5-example-scenario)
- [6. Key Algorithms](#6-key-algorithms)
- [7. Performance Optimization](#7-performance-optimization)

---

## 1. Overview

This PIM (Processing-In-Memory) simulator accurately predicts the performance of eFlash-based Neural Network Accelerators. It adopts an Event-Driven Simulation approach to precisely model hardware parallelism, data transfers, and resource contention at the nanosecond level.

### Key Features
- **Event-Driven Architecture**: Process events in chronological order for accurate timing simulation
- **Transfer-Compute Separation**: Model data transfer and computation independently for maximum parallelism
- **Multi-location Storage**: Duplicate activations across multiple SRAMs to minimize transfer time
- **Location-Specific Reference Counting**: Fine-grained memory management per storage location

---

## 2. System Architecture

### 2.1 Hardware Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                     PIM Simulator                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Shared SRAM (10-20 MB)                                      │
│  • Shared across all Arrays                                  │
│  • Bandwidth: 10 GB/s                                        │
│  • Used for inter-array data transfer                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Array 0     │  │  Array 1     │  │  Array 2     │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ 8 Areas      │  │ 8 Areas      │  │ 8 Areas      │      │
│  │              │  │              │  │              │      │
│  │ Internal     │  │ Internal     │  │ Internal     │      │
│  │ SRAM (1-2MB) │  │ SRAM (1-2MB) │  │ SRAM (1-2MB) │      │
│  │ • Dedicated  │  │ • Dedicated  │  │ • Dedicated  │      │
│  │ • Zero       │  │ • Zero       │  │ • Zero       │      │
│  │   transfer   │  │   transfer   │  │   transfer   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Software Components

- **Event Queue**: Min-heap for time-ordered event management
- **State Tracking**: 
  - `array_busy_until`: Array availability time
  - `activation_lifetimes`: Activation location and usage info
  - `running_nodes`: Currently executing nodes
  - `completed_nodes`: Completed node list
- **Statistics**: Transfer time, compute time measurement

---

## 3. Core Concepts

### 3.1 Event-Driven Simulation

The simulator does not advance time continuously but only processes discrete event occurrence points.

**Event Types:**
- `TRANSFER_START`: Data transfer begins
- `TRANSFER_DONE`: Data transfer completes
- `COMPUTE_START`: Computation begins
- `COMPUTE_DONE`: Computation completes

**Principle:**
```
Time 0 ──→ Event@100 ──→ Event@200 ──→ Event@80381.6 ──→
   │           │            │               │
 Start     Compute1      Compute2      Transfer
          Complete      Complete       Complete
```

Intermediate times (e.g., t=50, t=150) are skipped as no events occur.

### 3.2 Transfer-Compute Separation

**Previous Approach (Incorrect):**
```
Compute Start → [Data Transfer + Computation] → Compute Done
                 (Single block processing)
```

**Improved Approach (Current):**
```
Schedule → TRANSFER_START → TRANSFER_DONE → COMPUTE_START → COMPUTE_DONE
              ↓                 ↓                ↓               ↓
         Transfer Begin    Transfer End    Compute Begin   Compute End
```

**Benefits:**
1. **Parallelism**: Data transfer executes independently of Array computation
2. **Accuracy**: Clear separation and measurement of transfer vs compute time
3. **Optimization**: Data transfer can start even when Array is busy

### 3.3 Multi-location Storage

Store a single activation in multiple SRAMs to minimize transfer time.

**Example:**
```
conv1_output needed by:
  - conv2a (runs on Array 0)
  - conv2b (runs on Array 1)

Storage decision:
  - Array 0 Internal SRAM: for conv2a (0 ns transfer)
  - Shared SRAM: for conv2b (80 ms transfer)

┌────────────────┐
│ conv1_output   │
│ Size: 784 KB   │
└────────────────┘
     ↓       ↓
     ↓       └──────────────┐
     ↓                      ↓
┌──────────┐         ┌──────────┐
│ Array 0  │         │ Shared   │
│ SRAM     │         │ SRAM     │
│          │         │          │
│ conv2a   │         │ conv2b   │
│ (0 ns)   │         │ (80 ms)  │
└──────────┘         └──────────┘
```

### 3.4 Location-Specific Reference Counting

Maintain independent reference counts per storage location for immediate deallocation.

**Structure:**
```python
activation_lifetimes["conv1_output"] = {
    'buffer': ActivationBuffer(size=784KB),
    'storage_locations': {
        'array_0_sram': {
            'consumers': {'conv2a'},
            'ref_count': 1
        },
        'shared_sram': {
            'consumers': {'conv2b'},
            'ref_count': 1
        }
    }
}
```

**Deallocation Process:**
```
conv2a completes:
  → array_0_sram ref_count: 1 → 0
  → Deallocate from array_0_sram (free 784 KB)
  → shared_sram still retained (conv2b using)

conv2b completes:
  → shared_sram ref_count: 1 → 0
  → Deallocate from shared_sram
  → All locations deallocated → remove lifetime entry
```

---

## 4. Execution Flow

### 4.1 Initialization

```
[Step 1] Initialize Simulator
  ↓
• Create Event Queue (empty Min-Heap)
• Initialize Array state (array_busy_until = {})
• Initialize Memory (Shared SRAM, Array SRAMs)
  ↓
[Step 2] Analyze Compute Graph
  ↓
• Identify inter-node dependencies
• Find source nodes (nodes with no dependencies)
  ↓
[Step 3] Prepare Input Data
  ↓
• Create input buffer
• Allocate to Shared SRAM
  ↓
[Step 4] Schedule Source Nodes
  ↓
• Call _schedule_compute(node, t=0) for each source
• Initial events added to Event Queue
```

### 4.2 Node Scheduling (_schedule_compute)

```python
_schedule_compute(node, dependency_completion_time)
```

**Process:**

```
1. Calculate Input Data Transfer Time
   ────────────────────────────────────
   FOR EACH input node:
     • Find optimal location for input activation
       - If in internal SRAM: 0 ns transfer
       - If only in shared SRAM: calculate transfer time
     • Accumulate to transfer_time_ns

2. Generate Events (Branch on Transfer Need)
   ────────────────────────────────────
   IF transfer_time_ns > 0:
     ┌─────────────────────────────────────────┐
     │ Case A: Transfer Required                │
     ├─────────────────────────────────────────┤
     │ • CREATE TRANSFER_START event           │
     │   - Time: dependency_completion_time    │
     │   - Starts immediately (Array-agnostic) │
     │                                         │
     │ • CREATE TRANSFER_DONE event            │
     │   - Time: start + transfer_time         │
     │   - Will trigger COMPUTE events later   │
     └─────────────────────────────────────────┘
   ELSE:
     ┌─────────────────────────────────────────┐
     │ Case B: No Transfer (Internal SRAM only)│
     ├─────────────────────────────────────────┤
     │ • Check Array availability              │
     │   compute_start = MAX(dep_time, busy)   │
     │                                         │
     │ • CREATE COMPUTE_START event            │
     │ • CREATE COMPUTE_DONE event             │
     │ • Update array_busy_until               │
     └─────────────────────────────────────────┘

3. Mark Node as Running
```

### 4.3 Event Loop

```
Main Event Loop
────────────────────────────────────────
WHILE Event Queue NOT EMPTY:
  
  [1] Extract earliest event from Queue
      • Pop from Min-Heap (O(log n))
      • Update simulation time
  
  [2] Identify event type
  
  [3] Dispatch to appropriate handler:
      ┌──────────────┬──────────────┬──────────────┬──────────────┐
      │TRANSFER_START│TRANSFER_DONE │COMPUTE_START │COMPUTE_DONE  │
      └──────────────┴──────────────┴──────────────┴──────────────┘
           ↓              ↓              ↓              ↓
        Handler        Handler        Handler        Handler
  
  [4] Handler performs required operations:
      • Update state
      • Log information
      • Generate next events (chaining)
  
  Continue to next event
```

### 4.4 Key Event Handlers

#### TRANSFER_START Handler
```python
_handle_transfer_start(event)
```
- Record transfer start time
- Log transfer details (buffer, location, time)

#### TRANSFER_DONE Handler (Critical!)
```python
_handle_transfer_done(event)
```

**Process:**
```
1. Record transfer completion time

2. Calculate Compute Start Time
   ────────────────────────────────────
   array_available = array_busy_until[array_id]
   compute_start = MAX(transfer_done_time, array_available)
   
   • Transfer is complete, but
   • If Array is still busy, wait
   • If both ready, start immediately

3. Generate COMPUTE Events (HERE!)
   ────────────────────────────────────
   • CREATE COMPUTE_START event
   • CREATE COMPUTE_DONE event
   • Update array_busy_until
```

#### COMPUTE_START Handler
```python
_handle_compute_start(event)
```
- Record computation start time
- Log computation start

#### COMPUTE_DONE Handler (Most Complex!)
```python
_handle_compute_done(event)
```

**Process:**
```
Phase 1: Node Completion
────────────────────────────────────
• Mark node as complete
• Add to completed_nodes
• Remove from running_nodes

Phase 2: Deallocate Input Activations (Location-Specific)
────────────────────────────────────
FOR EACH input node:
  • Query input activation
  • Find optimal location (where it was read from)
  • Decrement reference count at that location
  • IF ref_count == 0:
      - Deallocate memory at that location
      - Remove from storage_locations
  • IF all locations deallocated:
      - Remove from activation_lifetimes

Phase 3: Create and Allocate Output Activation
────────────────────────────────────
[1] Create output buffer metadata
[2] Identify consumer nodes
[3] Determine storage locations (Multi-location)
    
    _determine_activation_locations(producer, consumers)
    ↓
    IF all consumers in same Array:
      → Internal SRAM only
    ELSE IF consumers in different Array only:
      → Shared SRAM only
    ELSE IF both:
      → Internal SRAM + Shared SRAM (duplicate)

[4] Allocate memory at each location
    FOR EACH location:
      • Get SRAM object
      • Call allocate(buffer_id, size)
      • Store consumers for this location
      • Set reference count

[5] Register lifetime information

Phase 4: Schedule Next Nodes
────────────────────────────────────
_schedule_ready_nodes(current_time)
  ↓
  • Find ready nodes (all dependencies satisfied)
  • Call _schedule_compute for each ready node
  • New events added to Queue
```

---

## 5. Example Scenario

### 5.1 Compute Graph
```
Input
  ↓
conv1 (Array 0)
  ↓
  ├─→ conv2a (Array 0)
  └─→ conv2b (Array 1)
```

### 5.2 Execution Timeline

```
t=0 ns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: COMPUTE_START conv1
• conv1 computation starts on Array 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=100 ns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: COMPUTE_DONE conv1

Actions:
  [1] Complete conv1
  [2] Deallocate inputs: None (input node)
  [3] Create output: conv1_output (784 KB)
      
      Consumer analysis:
        • conv2a (Array 0) - same Array
        • conv2b (Array 1) - different Array
      
      Storage decision:
        • Array 0 Internal SRAM (for conv2a)
        • Shared SRAM (for conv2b)
      
      Memory allocation:
        ┌─────────────────────────────────────┐
        │ conv1_output (784 KB)               │
        ├─────────────────────────────────────┤
        │ • array_0_sram: [conv2a] ref=1      │
        │ • shared_sram: [conv2b] ref=1       │
        └─────────────────────────────────────┘
  
  [4] Schedule ready nodes:
      
      conv2a scheduling:
        • Input: conv1_output from array_0_sram (0 ns)
        • transfer_time = 0
        → COMPUTE_START(100), COMPUTE_DONE(200)
      
      conv2b scheduling:
        • Input: conv1_output from shared_sram (80,281.6 ns)
        • transfer_time = 80,281.6 ns
        → TRANSFER_START(100), TRANSFER_DONE(80,381.6)

Event Queue:
  [100] COMPUTE_START conv2a
  [100] TRANSFER_START conv2b
  [200] COMPUTE_DONE conv2a
  [80381.6] TRANSFER_DONE conv2b

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=100 ns (Simultaneous Events)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: COMPUTE_START conv2a
• conv2a computation starts (Array 0)
• Input data already in array_0_sram
• 0 ns transfer time

Event: TRANSFER_START conv2b
• conv2b data transfer starts
• Transfer from Shared SRAM to Array 1
• Transferring... (80,281.6 ns duration)

Current State:
  • Array 0: conv2a computing
  • Transfer: conv2b data transferring (Array-independent)
  • PARALLEL EXECUTION! ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=200 ns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: COMPUTE_DONE conv2a

Actions:
  [1] Complete conv2a
  [2] Deallocate inputs:
      • conv1_output from array_0_sram
      • ref_count: 1 → 0
      • Deallocate from array_0_sram (free 784 KB)
      • shared_sram still retained (conv2b using)

Memory State:
  conv1_output remains only in shared_sram

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=80,381.6 ns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: TRANSFER_DONE conv2b
• Transfer complete
• Generate COMPUTE events:
  → COMPUTE_START(80,381.6)
  → COMPUTE_DONE(80,481.6)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=80,381.6 ns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: COMPUTE_START conv2b
• conv2b computation starts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=80,481.6 ns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event: COMPUTE_DONE conv2b

Actions:
  [1] Complete conv2b
  [2] Deallocate inputs:
      • conv1_output from shared_sram
      • ref_count: 1 → 0
      • Deallocate from shared_sram
      • All locations deallocated → remove from lifetimes

Final Result:
  • Total time: 80,481.6 ns
  • Transfer time: 80,281.6 ns (conv2b only)
  • Compute time: 300 ns (conv1: 100, conv2a: 100, conv2b: 100)
```

---

## 6. Key Algorithms

### 6.1 Storage Location Determination

```python
def _determine_activation_locations(producer, consumers):
    producer_array = producer.array_id
    consumer_arrays = set(c.array_id for c in consumers)
    
    if not consumers:
        return ["shared_sram"]
    
    if len(consumer_arrays) == 1 and producer_array in consumer_arrays:
        # All consumers in same array → Internal SRAM only
        return [f"array_{producer_array}_sram"]
    
    if len(consumer_arrays) == 1 and producer_array not in consumer_arrays:
        # All consumers in different array → Shared SRAM only
        return ["shared_sram"]
    
    # Multiple arrays → Duplicate storage
    locations = []
    if producer_array in consumer_arrays:
        locations.append(f"array_{producer_array}_sram")
    if any(c.array_id != producer_array for c in consumers):
        locations.append("shared_sram")
    
    return locations
```

### 6.2 Optimal Location Selection

```python
def _get_optimal_location(buffer_id, consumer_array_id):
    """
    Determine optimal location for consumer to read from
    """
    internal_location = f"array_{consumer_array_id}_sram"
    
    if internal_location in storage_locations and ref_count > 0:
        # Prefer internal SRAM if available
        return internal_location
    else:
        # Fall back to shared SRAM
        return "shared_sram"
```

### 6.3 Transfer Time Calculation

```python
def _calculate_transfer_time(size_bytes):
    """
    Calculate transfer time for shared SRAM access
    """
    BANDWIDTH = 10 * 1024 * 1024 * 1024  # 10 GB/s
    transfer_time_ns = (size_bytes / BANDWIDTH) * 1e9
    return transfer_time_ns
```

### 6.4 Consumers per Location

```python
def _get_consumers_for_location(consumers, location, producer_array):
    """
    Filter consumers that will use this specific location
    """
    if location == "shared_sram":
        # Shared SRAM: only consumers from different arrays
        return {c for c in consumers if c.array_id != producer_array}
    else:
        # Internal SRAM: only consumers from same array
        array_id = extract_array_id(location)
        return {c for c in consumers if c.array_id == array_id}
```

---

## 7. Performance Optimization

### 7.1 Parallelism Maximization

**Transfer-Compute Independence:**
- Data transfer can proceed independently of Array computation
- Transfer can start even when target Array is busy
- Overlap transfer time with other Array computations

**Example:**
```
Array 0: [====conv1====][====conv2a====]
Transfer:      [===========conv2b transfer===========]
Array 1:                                        [==conv2b compute==]
         ↑
         Transfer starts while Array 0 is busy!
```

### 7.2 Memory Efficiency

**Location-Specific Deallocation:**
- Free memory from each location as soon as no longer needed
- Don't wait for all consumers to complete
- Immediate memory reclamation

**Example:**
```
t=200: conv2a done → Free array_0_sram (784 KB available)
t=80481: conv2b done → Free shared_sram (784 KB available)

vs Previous: Wait until t=80481 to free everything
```

### 7.3 Transfer Minimization

**Multi-location Storage:**
- Duplicate activations strategically
- Use internal SRAM whenever possible (0 ns transfer)
- Only use shared SRAM for cross-array communication

**Savings:**
```
Without duplication: All consumers read from shared SRAM
  conv2a: 80,281.6 ns transfer
  conv2b: 80,281.6 ns transfer
  Total: 160,563.2 ns

With duplication: Internal SRAM for same-array access
  conv2a: 0 ns transfer (internal SRAM)
  conv2b: 80,281.6 ns transfer (shared SRAM)
  Total: 80,281.6 ns
  
Savings: 50% transfer time reduction!
```

---

## 8. Implementation Details

### 8.1 Event Queue Implementation
- **Data Structure**: Min-Heap (Python `heapq`)
- **Complexity**: O(log n) insertion, O(log n) extraction
- **Ordering**: Events sorted by `time_ns` field

### 8.2 State Management
```python
# Core state variables
self.event_queue = []                    # Min-heap of events
self.current_time_ns = 0                 # Simulation time
self.array_busy_until = {}               # {array_id: time}
self.running_nodes = {}                  # {node_id: (array, area)}
self.completed_nodes = []                # List of completed nodes
self.activation_lifetimes = {}           # Activation metadata

# Statistics
self.total_transfer_time_ns = 0
self.total_compute_time_ns = 0
```

### 8.3 Activation Lifetime Structure
```python
activation_lifetimes[buffer_id] = {
    'buffer': ActivationBuffer,          # Metadata
    'storage_locations': {               # Multi-location support
        'array_0_sram': {
            'sram': SRAMBuffer,          # SRAM object
            'consumers': {'node1', ...}, # Consumers using this location
            'ref_count': 1               # Reference count
        },
        'shared_sram': {
            'sram': SRAMBuffer,
            'consumers': {'node2', ...},
            'ref_count': 1
        }
    }
}
```

---

## 9. Summary

This PIM simulator accurately models hardware behavior through:

1. **Event-Driven Architecture**: Precise timing simulation at nanosecond granularity
2. **Transfer-Compute Separation**: Maximize parallelism through independent modeling
3. **Multi-location Storage**: Minimize transfer time through strategic duplication
4. **Location-Specific Memory Management**: Optimize memory usage through fine-grained deallocation

The simulator enables accurate performance prediction for eFlash-based PIM accelerators, supporting design space exploration and optimization before hardware implementation.

---

## Appendix: Event Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Event Flow Summary                        │
└─────────────────────────────────────────────────────────────┘

_schedule_compute(node, t)
  ↓
Calculate transfer_time
  ↓
IF transfer_time > 0:
  ┌──────────────────────────────────────────────────────┐
  │ TRANSFER_START(t)                                    │
  │   ↓                                                  │
  │ TRANSFER_DONE(t + Δt)                                │
  │   ↓                                                  │
  │ _handle_transfer_done()                              │
  │   ↓                                                  │
  │ COMPUTE_START(max(t+Δt, array_busy))                 │
  │   ↓                                                  │
  │ COMPUTE_DONE(t + Δt + compute_time)                  │
  └──────────────────────────────────────────────────────┘
ELSE:
  ┌──────────────────────────────────────────────────────┐
  │ COMPUTE_START(max(t, array_busy))                    │
  │   ↓                                                  │
  │ COMPUTE_DONE(t + compute_time)                       │
  └──────────────────────────────────────────────────────┘
  ↓
_handle_compute_done()
  ↓
1. Deallocate inputs (location-specific)
2. Allocate outputs (multi-location)
3. Schedule ready nodes
  ↓
Next iteration...
```
