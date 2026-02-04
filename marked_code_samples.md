```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/prometheus/Prometheus/util/logging Below is the code repository: github.com/prometheus/Prometheus/ Below is the imported package path "context"; "log/slog"; "sync"; "time" The receiver struct definitions of the function is type Deduper struct {
	next   *slog.Logger
	repeat time.Duration
	<ATTN>quit</ATTN>   chan struct{}
	<ATTN>mtx</ATTN>    *sync.RWMutex
	<ATTN>seen</ATTN>   map[string]time.Time
}

// Methods:
- func (d *Deduper) Enabled((ctx context.Context, level slog.Level)) bool
- func (d *Deduper) Handle((ctx context.Context, r slog.Record)) error
- func (d *Deduper) WithAttrs(attrs *ast.ArrayType) slog.Handler
- func (d *Deduper) WithGroup(name string) slog.Handler
- func (d *Deduper) Stop()
- func (d *Deduper) run()
 The parameter struct definition or not exist of the function is attrs *ast.ArrayType The return value struct definitions or not exist of the function is slog.Handler The code snippets before the function is func Dedupe(next *slog.Logger, repeat time.Duration) *Deduper {
	d := &Deduper{
		next:   next,
		repeat: repeat,
		quit:   make(chan struct{}),
		mtx:    new(sync.RWMutex),
		seen:   map[string]time.Time{},
	}
	go d.run()
	return d
}

func (d *Deduper) Enabled(ctx context.Context, level slog.Level) bool {
	return d.next.Enabled(ctx, level)
}

func (d *Deduper) Handle(ctx context.Context, r slog.Record) error {
	line := r.Message
	d.mtx.RLock()
	last, ok := d.seen[line]
	d.mtx.RUnlock()

	if ok && time.Since(last) < d.repeat {
		return nil
	}

	d.mtx.Lock()
	if len(d.seen) < maxEntries {
		d.seen[line] = time.Now()
	}
	d.mtx.Unlock()

	return d.next.Handler().Handle(ctx, r.Clone())
} And here is the function you are asked to complete func (d *Deduper) WithAttrs(attrs *ast.ArrayType) slog.Handler
func (d *Deduper) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &Deduper{
		next:   slog.New(d.next.Handler().WithAttrs(attrs)),
		repeat: d.repeat, <MID> 	}
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		quit:   d.quit,
		seen:   d.seen,
		mtx:    d.mtx,<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/cockroachdb/cockroach/pkg/kv/kvserver/multiqueue Below is the code repository: github.com/cockroachdb/cockroach/ Below is the imported package path "container/heap"; "github.com/cockroachdb/cockroach/pkg/util/syncutil"; "github.com/cockroachdb/errors"; "github.com/cockroachdb/redact" The receiver struct definitions of the function is type MultiQueue struct {
	<ATTN>mu</ATTN>               syncutil.Mutex
	concurrencyLimit int
	remainingRuns    int
	mapping          map[int]int
	lastQueueIndex   int
	outstanding      []notifyHeap
}

// Methods:
- func (m *MultiQueue) tryRunNextLocked()
- func (m *MultiQueue) UpdateConcurrencyLimit(newLimit int)
- func (m *MultiQueue) updateConcurrencyLimitLocked(newLimit int)
- func (m *MultiQueue) Add((queueType int, priority float64, maxQueueLength int64)) ((*Task, error))
- func (m *MultiQueue) Cancel(task *Task)
- func (m *MultiQueue) Release(permit *Permit)
- func (m *MultiQueue) releaseLocked(permit *Permit)
- func (m *MultiQueue) AvailableLen() int
- func (m *MultiQueue) QueueLen() int
- func (m *MultiQueue) queueLenLocked() int
- func (m *MultiQueue) MaxConcurrency() int
 The parameter struct definition or not exist of the function is newLimit int The return value struct definitions or not exist of the function is not exist The code snippets before the function is func (h *notifyHeap) Push(x interface{}) {
	t := x.(*Task)
	// Set the index to the end, it will be moved later
	t.heapIdx = h.Len()
	*h = append(*h, t)
}

func (h *notifyHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	old[n-1] = nil
	*h = old[0 : n-1]
	// No longer in the heap so clear the index
	x.heapIdx = -1

	return x
}

func (h *notifyHeap) tryRemove(task *Task) bool {
	if task.heapIdx < 0 {
		return false
	}
	heap.Remove(h, task.heapIdx)
	return true
}

func NewMultiQueue(maxConcurrency int) *MultiQueue {
	queue := MultiQueue{
		remainingRuns:    maxConcurrency,
		concurrencyLimit: maxConcurrency,
		mapping:          make(map[int]int),
	}
	queue.lastQueueIndex = -1

	return &queue
}

func (m *MultiQueue) tryRunNextLocked() {
	// If no permits are left, then we can't run anything.
	if m.remainingRuns <= 0 {
		return
	}

	for i := 0; i < len(m.outstanding); i++ {
		// Start with the next queue in order and iterate through all empty queues.
		// If all queues are empty then return, as there is nothing to run.
		index := (m.lastQueueIndex + i + 1) % len(m.outstanding)
		if m.outstanding[index].Len() > 0 {
			task := heap.Pop(&m.outstanding[index]).(*Task)
			task.permitC <- &Permit{valid: true}
			m.remainingRuns--
			m.lastQueueIndex = index
			return
		}
	}
} And here is the function you are asked to complete func (m *MultiQueue) UpdateConcurrencyLimit(newLimit int)
func (m *MultiQueue) UpdateConcurrencyLimit(newLimit int) { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	m.mu.Lock()
	defer m.mu.Unlock()
	m.updateConcurrencyLimitLocked(newLimit)<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/pkg/scheduler/metrics Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "time"; "k8s.io/component-base/metrics" The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is (bufferSize int, interval time.Duration, stopCh *ast.ChanType) The return value struct definitions or not exist of the function is *MetricAsyncRecorder

// Type MetricAsyncRecorder definition:
type MetricAsyncRecorder struct {
	// bufferCh is a channel that serves as a metrics buffer before the metricsRecorder goroutine reports it.
	<ATTN>bufferCh</ATTN> chan *histogramVecMetric
	// if bufferSize is reached, incoming metrics will be discarded.
	bufferSize int
	// how often the recorder runs to flush the metrics.
	interval time.Duration

	// aggregatedInflightEventMetric is only to record InFlightEvents metric asynchronously.
	// It's a map from gaugeVecMetricKey to the aggregated value
	// and the aggregated value is flushed to Prometheus every time the interval is reached.
	// Note that we don't lock the map deliberately because we assume the queue takes lock before updating the in-flight events.
	aggregatedInflightEventMetric              map[gaugeVecMetricKey]int
	aggregatedInflightEventMetricLastFlushTime time.Time
	aggregatedInflightEventMetricBufferCh      chan *gaugeVecMetric

	// stopCh is used to stop the goroutine which periodically flushes metrics.
	<ATTN>stopCh</ATTN> <-chan struct{}
	// IsStoppedCh indicates whether the goroutine is stopped. It's used in tests only to make sure
	// the metric flushing goroutine is stopped so that tests can collect metrics for verification.
	<ATTN>IsStoppedCh</ATTN> chan struct{}
}
// Methods:
- func (r *MetricAsyncRecorder) ObservePluginDurationAsync((extensionPoint string, pluginName string, status string, value float64))
- func (r *MetricAsyncRecorder) ObserveQueueingHintDurationAsync((pluginName string, event string, hint string, value float64))
- func (r *MetricAsyncRecorder) ObserveInFlightEventsAsync((eventLabel string, valueToAdd float64, forceFlush bool))
- func (r *MetricAsyncRecorder) observeMetricAsync((m *metrics.HistogramVec, value float64, labelsValues *ast.Ellipsis))
- func (r *MetricAsyncRecorder) run()
- func (r *MetricAsyncRecorder) FlushMetrics()
 The code snippets before the function is func NewBackoffPodsRecorder() *PendingPodsRecorder {
	return &PendingPodsRecorder{
		recorder: BackoffPods(),
	}
}

func NewGatedPodsRecorder() *PendingPodsRecorder {
	return &PendingPodsRecorder{
		recorder: GatedPods(),
	}
}

func (r *PendingPodsRecorder) Inc() {
	r.recorder.Inc()
}

func (r *PendingPodsRecorder) Dec() {
	r.recorder.Dec()
}

func (r *PendingPodsRecorder) Clear() {
	r.recorder.Set(float64(0))
} And here is the function you are asked to complete func NewMetricsAsyncRecorder((bufferSize int, interval time.Duration, stopCh *ast.ChanType)) *MetricAsyncRecorder
func NewMetricsAsyncRecorder(bufferSize int, interval time.Duration, stopCh <-chan struct{}) *MetricAsyncRecorder {
	recorder := &MetricAsyncRecorder{
		bufferCh:                      make(chan *histogramVecMetric, bufferSize),
		bufferSize:                    bufferSize,
		interval:                      interval,
		stopCh:                        stopCh,
		aggregatedInflightEventMetric: make(map[gaugeVecMetricKey]int),
		aggregatedInflightEventMetricLastFlushTime: time.Now(),
		aggregatedInflightEventMetricBufferCh:      make(chan *gaugeVecMetric, bufferSize),
		IsStoppedCh:                                make(chan struct{}), <MID> 	return recorder
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	}
	go recorder.run()<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/syncthing/syncthing Below is the code repository: github.com/syncthing/syncthing/ Below is the imported package path "archive/tar"; "archive/zip"; "bytes"; "compress/flate"; "compress/gzip"; "encoding/json"; "errors"; "flag"; "fmt"; "io"; "log"; "os"; "os/exec"; "os/user"; "path/filepath"; "regexp"; "runtime"; "strconv"; "strings"; "text/template"; "time"; "github.com/syncthing/syncthing/lib/build"; "github.com/syncthing/syncthing/lib/upgrade"; "sigs.k8s.io/yaml" The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is not exist The code snippets before the function is func rebuildAssets() {
	os.Setenv("SOURCE_DATE_EPOCH", fmt.Sprint(buildStamp()))
	runPrint(goCmd, "generate", "github.com/syncthing/syncthing/lib/api/auto", "github.com/syncthing/syncthing/cmd/infra/strelaypoolsrv/auto")
}

func lazyRebuildAssets() {
	shouldRebuild := shouldRebuildAssets("lib/api/auto/gui.files.go", "gui") ||
		shouldRebuildAssets("cmd/infra/strelaypoolsrv/auto/gui.files.go", "cmd/infra/strelaypoolsrv/gui")

	if shouldRebuild {
		rebuildAssets()
	}
}

func shouldRebuildAssets(target, srcdir string) bool {
	info, err := os.Stat(target)
	if err != nil {
		// If the file doesn't exist, we must rebuild it
		return true
	}

	// Check if any of the files in gui/ are newer than the asset file. If
	// so we should rebuild it.
	currentBuild := info.ModTime()
	assetsAreNewer := false
	stop := errors.New("no need to iterate further")
	filepath.Walk(srcdir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.ModTime().After(currentBuild) {
			assetsAreNewer = true
			return stop
		}
		return nil
	})

	return assetsAreNewer
}

func updateDependencies() {
	// Figure out desired Go version
	bs, err := os.ReadFile("go.mod")
	if err != nil {
		log.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^go\s+([0-9.]+)`)
	matches := re.FindSubmatch(bs)
	if len(matches) != 2 {
		log.Fatal("failed to parse go.mod")
	}
	goVersion := string(matches[1])

	runPrint(goCmd, "get", "-u", "./...")
	runPrint(goCmd, "mod", "tidy", "-go="+goVersion, "-compat="+goVersion)

	// We might have updated the protobuf package and should regenerate to match.
	proto()
}

func proto() {
	// buf needs to be installed
	// https://buf.build/docs/installation/
	runPrint("buf", "generate")
} And here is the function you are asked to complete func testmocks()
func <ATTN>testmocks</ATTN>() {
	<ATTN>args</ATTN> := <ATTN>[]string</ATTN>{
		"generate",
		"github.com/syncthing/syncthing/lib/config",
		"github.com/syncthing/syncthing/lib/connections",
		"github.com/syncthing/syncthing/lib/discover",
		"github.com/syncthing/syncthing/lib/events",
		"github.com/syncthing/syncthing/lib/model",
		"github.com/syncthing/syncthing/lib/protocol", <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	}
	runPrint(goCmd, args...)<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/cockroachdb/cockroach/pkg/util/mon Below is the code repository: github.com/cockroachdb/cockroach/ Below is the imported package path "context"; "fmt"; "io"; "math"; "strconv"; "strings"; "unsafe"; "github.com/cockroachdb/cockroach/pkg/settings"; "github.com/cockroachdb/cockroach/pkg/settings/cluster"; "github.com/cockroachdb/cockroach/pkg/util"; "github.com/cockroachdb/cockroach/pkg/util/buildutil"; "github.com/cockroachdb/cockroach/pkg/util/envutil"; "github.com/cockroachdb/cockroach/pkg/util/humanizeutil"; "github.com/cockroachdb/cockroach/pkg/util/ioctx"; "github.com/cockroachdb/cockroach/pkg/util/log"; "github.com/cockroachdb/cockroach/pkg/util/log/logcrash"; "github.com/cockroachdb/cockroach/pkg/util/metric"; "github.com/cockroachdb/cockroach/pkg/util/syncutil"; "github.com/cockroachdb/cockroach/pkg/util/uuid"; "github.com/cockroachdb/errors"; "github.com/cockroachdb/redact"; "github.com/dustin/go-humanize" The receiver struct definitions of the function is type BytesMonitor struct {
	mu struct {
		syncutil.Mutex

		// curAllocated tracks the current amount of bytes allocated at this
		// monitor by its client components.
		curAllocated int64

		// maxAllocated tracks the high water mark of allocations. Used for
		// monitoring.
		maxAllocated int64

		// curBudget represents the budget allocated at the pool on behalf of
		// this monitor.
		curBudget BoundAccount

		//  Both fields below are protected by the mutex because they might be
		//  updated after the monitor has been instantiated.

		// curBytesCount is the metric object used to track number of bytes
		// reserved by the monitor during its lifetime.
		curBytesCount *metric.Gauge

		// maxBytesHist is the metric object used to track the high watermark of bytes
		// allocated by the monitor during its lifetime.
		maxBytesHist metric.IHistogram

		// head is the head of the doubly-linked list of children of this
		// monitor.
		// NOTE: a child's mutex **cannot** be acquired while holding this
		// monitor's lock since it could lead to deadlocks. The main allocation
		// code path (reserveBytes() and releaseBytes()) might acquire the
		// parent's lock, so only locking "upwards" is allowed while keeping the
		// current monitor's lock.
		head *BytesMonitor

		// stopped indicates whether this monitor has been stopped.
		stopped bool

		// relinquishAllOnReleaseBytes, if true, indicates that the monitor should
		// relinquish all bytes on releaseBytes() call.
		// NB: this field doesn't need mutex protection but is inside of mu
		// struct in order to reduce the struct size.
		relinquishAllOnReleaseBytes bool

		// tracksDisk indicates whether this monitor is for tracking disk
		// resources.
		// NB: this field doesn't need mutex protection but is inside of mu
		// struct in order to reduce the struct size.
		tracksDisk bool

		// rootSQLMonitor indicates whether this monitor is the root SQL memory
		// monitor (in which case, memory budget exceeded errors should have a
		// hint to increase --max-sql-memory parameter).
		// NB: this field doesn't need mutex protection but is inside of mu
		// struct in order to reduce the struct size.
		rootSQLMonitor bool

		// longLiving indicates whether lifetime of this monitor matches the
		// server's life, and as such, this monitor is exempted from having to
		// be stopped when its ancestor monitor is stopped.
		// NB: this field doesn't need mutex protection but is inside of mu
		// struct in order to reduce the struct size.
		<ATTN>longLiving</ATTN> bool
	}

	// parentMu encompasses the fields that must be accessed while holding the
	// mutex of the parent monitor.
	parentMu struct {
		// prevSibling and nextSibling are references to the previous and the
		// next siblings of this monitor (i.e. the previous and the next nodes
		// in the doubly-linked list of children of the parent monitor).
		prevSibling, nextSibling *BytesMonitor
	}

	// name identifies this monitor in logging messages.
	name Name

	// reserved indicates how many bytes were already reserved for this
	// monitor before it was instantiated. Allocations registered to
	// this monitor are first deducted from this budget. If there is no
	// pool, reserved determines the maximum allocation capacity of this
	// monitor. The reserved bytes are released to their owner monitor
	// upon Stop.
	reserved *BoundAccount

	// limit specifies a hard limit on the number of bytes a monitor allows to
	// be allocated. Note that this limit will not be observed if allocations
	// hit constraints on the owner monitor. This is useful to limit allocations
	// when an owner monitor has a larger capacity than wanted but should still
	// keep track of allocations made through this monitor. Note that child
	// monitors are affected by this limit.
	//
	// limit is computed from configLimit, the parent monitor and the
	// reserved budget during Start().
	limit int64

	// configLimit is the limit configured when the monitor is created.
	configLimit int64

	// poolAllocationSize specifies the allocation unit for requests to the
	// pool.
	poolAllocationSize int64

	settings *cluster.Settings
}

// Methods:
- func (mm *BytesMonitor) TraverseTree(monitorStateCb *ast.FuncType) error
- func (mm *BytesMonitor) traverseTree((level int, monitorStateCb *ast.FuncType)) error
- func (mm *BytesMonitor) MarkAsRootSQLMonitor()
- func (mm *BytesMonitor) StartNoReserved((ctx context.Context, pool *BytesMonitor))
- func (mm *BytesMonitor) Start((ctx context.Context, pool *BytesMonitor, reserved *BoundAccount))
- func (mm *BytesMonitor) EmergencyStop(ctx context.Context)
- func (mm *BytesMonitor) Stop(ctx context.Context)
- func (mm *BytesMonitor) Name() Name
- func (mm *BytesMonitor) Limit() int64
- func (mm *BytesMonitor) MarkLongLiving()
- func (mm *BytesMonitor) doStop((ctx context.Context, check bool))
- func (mm *BytesMonitor) MaximumBytes() int64
- func (mm *BytesMonitor) AllocBytes() int64
- func (mm *BytesMonitor) SetMetrics((curCount *metric.Gauge, maxHist metric.IHistogram))
- func (mm *BytesMonitor) Resource() Resource
- func (mm *BytesMonitor) MakeBoundAccount() BoundAccount
- func (mm *BytesMonitor) MakeEarmarkedBoundAccount() EarmarkedBoundAccount
- func (mm *BytesMonitor) MakeConcurrentBoundAccount() *ConcurrentBoundAccount
- func (mm *BytesMonitor) TransferAccount((ctx context.Context, origAccount *BoundAccount)) ((newAccount BoundAccount, err error))
- func (mm *BytesMonitor) makeBudgetExceededError(minExtra int64) error
- func (mm *BytesMonitor) reserveBytes((ctx context.Context, x int64)) error
- func (mm *BytesMonitor) releaseBytes((ctx context.Context, sz int64))
- func (mm *BytesMonitor) releaseBytesLocked((ctx context.Context, sz int64))
- func (mm *BytesMonitor) increaseBudget((ctx context.Context, minExtra int64)) error
- func (mm *BytesMonitor) roundSize(sz int64) int64
- func (mm *BytesMonitor) releaseBudget(ctx context.Context)
- func (mm *BytesMonitor) RelinquishAllOnReleaseBytes()
- func (mm *BytesMonitor) adjustBudget(ctx context.Context)
 The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is not exist The code snippets before the function is func NewUnlimitedMonitor(ctx context.Context, args Options) *BytesMonitor {
	if log.V(2) {
		log.Dev.InfofDepth(ctx, 1, "%s: starting unlimited monitor", args.Name)
	}
	// The limit is expected to not be set, but let's be conservative.
	args.Limit = math.MaxInt64
	m := NewMonitor(args)
	m.reserved = NewStandaloneBudget(math.MaxInt64)
	return m
}

func (mm *BytesMonitor) EmergencyStop(ctx context.Context) {
	mm.doStop(ctx, false)
}

func (mm *BytesMonitor) Stop(ctx context.Context) {
	mm.doStop(ctx, true)
}

func (mm *BytesMonitor) Name() Name {
	return mm.name
}

func (mm *BytesMonitor) Limit() int64 {
	return mm.limit
} And here is the function you are asked to complete func (mm *BytesMonitor) MarkLongLiving()
func (mm *BytesMonitor) MarkLongLiving() { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	mm.mu.Lock()
	defer mm.mu.Unlock()
	mm.mu.longLiving = true<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/influxdata/influxdb/influxql/query Below is the code repository: github.com/influxdata/influxdb/ Below is the imported package path "container/heap"; "context"; "io"; "sort"; "sync"; "time"; "github.com/influxdata/influxql"; "google.golang.org/protobuf/proto" The receiver struct definitions of the function is type integerIteratorScanner struct {
	input        *bufIntegerIterator
	err          error
	keys         []influxql.VarRef
	defaultValue interface{}
}

// Methods:
- func (s *integerIteratorScanner) Peek() ((int64, string, Tags))
- func (s *integerIteratorScanner) ScanAt((ts int64, name string, tags Tags, m *ast.MapType))
- func (s *integerIteratorScanner) useDefaults(m *ast.MapType)
- func (s *integerIteratorScanner) Stats() IteratorStats
- func (s *integerIteratorScanner) Err() error
- func (s *integerIteratorScanner) Close() error
 The parameter struct definition or not exist of the function is (ts int64, name string, tags Tags, m *ast.MapType) The return value struct definitions or not exist of the function is not exist The code snippets before the function is func (h *integerSortedMergeHeap) Less(i, j int) bool {
	x, y := h.items[i].point, h.items[j].point

	if h.opt.Ascending {
		if x.Name != y.Name {
			return x.Name < y.Name
		} else if xTags, yTags := x.Tags.Subset(h.opt.Dimensions), y.Tags.Subset(h.opt.Dimensions); !xTags.Equals(&yTags) {
			return xTags.ID() < yTags.ID()
		}

		if x.Time != y.Time {
			return x.Time < y.Time
		}

		if len(x.Aux) > 0 && len(x.Aux) == len(y.Aux) {
			for i := 0; i < len(x.Aux); i++ {
				v1, ok1 := x.Aux[i].(string)
				v2, ok2 := y.Aux[i].(string)
				if !ok1 || !ok2 {
					// Unsupported types used in Aux fields. Maybe they
					// need to be added here?
					return false
				} else if v1 == v2 {
					continue
				}
				return v1 < v2
			}
		}
		return false // Times and/or Aux fields are equal.
	}

	if x.Name != y.Name {
		return x.Name > y.Name
	} else if xTags, yTags := x.Tags.Subset(h.opt.Dimensions), y.Tags.Subset(h.opt.Dimensions); !xTags.Equals(&yTags) {
		return xTags.ID() > yTags.ID()
	}

	if x.Time != y.Time {
		return x.Time > y.Time
	}

	if len(x.Aux) > 0 && len(x.Aux) == len(y.Aux) {
		for i := 0; i < len(x.Aux); i++ {
			v1, ok1 := x.Aux[i].(string)
			v2, ok2 := y.Aux[i].(string)
			if !ok1 || !ok2 {
				// Unsupported types used in Aux fields. Maybe they
				// need to be added here?
				return false
			} else if v1 == v2 {
				continue
			}
			return v1 > v2
		}
	}
	return false // Times and/or Aux fields are equal.
}

func (h *integerSortedMergeHeap) Push(x interface{}) {
	h.items = append(h.items, x.(*integerSortedMergeHeapItem))
}

func (h *integerSortedMergeHeap) Pop() interface{} {
	old := h.items
	n := len(old)
	item := old[n-1]
	h.items = old[0 : n-1]
	return item
}

func newIntegerIteratorScanner(input IntegerIterator, keys []influxql.VarRef, defaultValue interface{}) *integerIteratorScanner {
	return &integerIteratorScanner{
		input:        newBufIntegerIterator(input),
		keys:         keys,
		defaultValue: defaultValue,
	}
}

func (s *integerIteratorScanner) Peek() (int64, string, Tags) {
	if s.err != nil {
		return ZeroTime, "", Tags{}
	}

	p, err := s.input.peek()
	if err != nil {
		s.err = err
		return ZeroTime, "", Tags{}
	} else if p == nil {
		return ZeroTime, "", Tags{}
	}
	return p.Time, p.Name, p.Tags
} And here is the function you are asked to complete func (s *integerIteratorScanner) ScanAt((ts int64, name string, tags Tags, m *ast.MapType))
func (s *integerIteratorScanner) ScanAt(ts int64, name string, tags Tags, m map[string]interface{}) {
	if s.err != nil {
		return
	}

	p, err := s.input.Next()
	if err != nil {
		s.err = err
		return
	} else if p == nil {
		s.useDefaults(m)
		return
	} else if p.Time != ts || p.Name != name || !p.Tags.Equals(&tags) {
		s.useDefaults(m)
		s.input.unread(p)
		return
	}

	if k := s.keys[0]; k.Val != "" {
		if <ATTN>p.Nil</ATTN> { <MID> 		}
	}
	for i, v := range p.Aux {
		k := s.keys[i+1]
		switch v.(type) {
		case float64, int64, uint64, string, bool:
			m[k.Val] = v
		default:
			// Insert the fill value if one was specified.
			if s.<ATTN>defaultValue</ATTN> != <ATTN>SkipDefault</ATTN> {
				m[k.Val] = <ATTN>castToType</ATTN>(s.defaultValue, k.Type)
			}
		}
	}
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
			if s.defaultValue != SkipDefault {
				m[k.Val] = castToType(s.defaultValue, k.Type)
			}
		} else {
			m[k.Val] = p.Value<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/olivere/elastic Below is the code repository: github.com/olivere/elastic/ Below is the imported package path not exist The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is FuzzyLikeThisQuery

// Type FuzzyLikeThisQuery definition:
type FuzzyLikeThisQuery struct {
	Query

	<ATTN>fields</ATTN>                 []string
	boost                  *float32
	likeText               *string
	fuzziness              interface{}
	prefixLength           *int
	maxQueryTerms          *int
	ignoreTF               *bool
	analyzer               string
	failOnUnsupportedField *bool
	queryName              string
}
// Methods:
- func (q FuzzyLikeThisQuery) Field(field string) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) Fields(fields *ast.Ellipsis) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) LikeText(likeText string) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) Fuzziness(fuzziness *ast.InterfaceType) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) PrefixLength(prefixLength int) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) MaxQueryTerms(maxQueryTerms int) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) IgnoreTF(ignoreTF bool) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) Analyzer(analyzer string) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) Boost(boost float32) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) FailOnUnsupportedField(fail bool) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) QueryName(queryName string) FuzzyLikeThisQuery
- func (q FuzzyLikeThisQuery) Source() *ast.InterfaceType
 The code snippets before the function is not exist And here is the function you are asked to complete func NewFuzzyLikeThisQuery() FuzzyLikeThisQuery
func <ATTN>NewFuzzyLikeThisQuery</ATTN>() FuzzyLikeThisQuery { <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	q := FuzzyLikeThisQuery{
		fields: make([]string, 0),
	}
	return q<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/olivere/elastic Below is the code repository: github.com/olivere/elastic/ Below is the imported package path not exist The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is parentType string The return value struct definitions or not exist of the function is HasParentFilter

// Type HasParentFilter definition:
type HasParentFilter struct {
	filter     Filter
	query      Query
	parentType string
	filterName string
	cache      *bool
	cacheKey   string
	innerHit   *InnerHit
}
// Methods:
- func (f HasParentFilter) Query(query Query) HasParentFilter
- func (f HasParentFilter) Filter(filter Filter) HasParentFilter
- func (f HasParentFilter) FilterName(filterName string) HasParentFilter
- func (f HasParentFilter) Cache(cache bool) HasParentFilter
- func (f HasParentFilter) CacheKey(cacheKey string) HasParentFilter
- func (f HasParentFilter) InnerHit(innerHit *InnerHit) HasParentFilter
- func (f HasParentFilter) Source() *ast.InterfaceType
 The code snippets before the function is not exist And here is the function you are asked to complete func NewHasParentFilter(parentType string) HasParentFilter
func NewHasParentFilter(parentType string) HasParentFilter {
	f := HasParentFilter{
		<ATTN>parentType</ATTN>: parentType, <MID> } Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
	}
	return f<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/kubernetes/Kubernetes/vendor/github.com/soheilhy/cmux Below is the code repository: github.com/kubernetes/Kubernetes/ Below is the imported package path "errors"; "fmt"; "io"; "net"; "sync"; "time" The receiver struct definitions of the function is not exist The parameter struct definition or not exist of the function is matchers *ast.ArrayType The return value struct definitions or not exist of the function is *ast.ArrayType The code snippets before the function is func (e ErrNotMatched) Timeout() bool { return false }

func (e errListenerClosed) Error() string   { return string(e) }

func (e errListenerClosed) Temporary() bool { return false }

func (e errListenerClosed) Timeout() bool   { return false }

func New(l net.Listener) CMux {
	return &cMux{
		root:        l,
		bufLen:      1024,
		errh:        func(_ error) bool { return true },
		donec:       make(chan struct{}),
		readTimeout: noTimeout,
	}
} And here is the function you are asked to complete func matchersToMatchWriters(matchers *ast.ArrayType) *ast.ArrayType
func matchersToMatchWriters(matchers []<ATTN>Matcher</ATTN>) []MatchWriter {
	mws := make([]MatchWriter, 0, len(matchers))
	<ATTN>for</ATTN> _, m := range <ATTN>matchers</ATTN> { <MID> 	}
	return mws
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
		cm := m
		mws = append(mws, func(w io.Writer, r io.Reader) bool {
			return cm(r)
		})<|im_end|>
```

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
This is a go programming task on some code contents. Given task: The task is to fill in the missing part of a go function according to the provided code   content. Below is the package path:github.com/cockroachdb/cockroach/pkg/sql/opt/xform Below is the code repository: github.com/cockroachdb/cockroach/ Below is the imported package path "bytes"; "fmt"; "sort"; "github.com/cockroachdb/cockroach/pkg/sql/opt"; "github.com/cockroachdb/cockroach/pkg/sql/opt/cycle"; "github.com/cockroachdb/cockroach/pkg/sql/opt/memo"; "github.com/cockroachdb/cockroach/pkg/sql/opt/props/physical"; "github.com/cockroachdb/cockroach/pkg/util/treeprinter"; "github.com/cockroachdb/errors"; "github.com/cockroachdb/redact" The receiver struct definitions of the function is type memoFormatter struct {
	buf              *bytes.Buffer
	flags            FmtFlags
	redactableValues bool

	o *Optimizer

	// groups reachable from the root, generated in breadth-first order.
	groups []group

	// groupIdx remembers the group index (in the groups slice) for the first
	// expression in each group.
	groupIdx map[opt.Expr]int
}

// Methods:
- func (mf *memoFormatter) format() string
- func (mf *memoFormatter) group(expr opt.Expr) int
- func (mf *memoFormatter) numberMemo(root opt.Expr)
- func (mf *memoFormatter) numberExpr(expr opt.Expr)
- func (mf *memoFormatter) populateStates()
- func (mf *memoFormatter) formatGroup(first memo.RelExpr)
- func (mf *memoFormatter) formatExpr(e opt.Expr)
- func (mf *memoFormatter) formatBest((best memo.RelExpr, required *physical.Required))
- func (mf *memoFormatter) formatPrivate((e opt.Expr, physProps *physical.Required))
- func (mf *memoFormatter) formatCycles() string
 The parameter struct definition or not exist of the function is not exist The return value struct definitions or not exist of the function is string The code snippets before the function is func (mf *memoFormatter) formatGroup(first memo.RelExpr) {
	for member := first; member != nil; member = member.NextExpr() {
		if member != first {
			mf.buf.WriteByte(' ')
		}
		mf.formatExpr(member)
	}
}

func (mf *memoFormatter) formatExpr(e opt.Expr) {
	if mf.redactableValues && opt.IsConstValueOp(e) && e.Private() == nil {
		// This marks FalseOp and TrueOp as redactable, e.g. instead of (true) we
		// print (‹true›).
		redact.Fprintf(mf.buf, "(%s", e.Op())
	} else {
		fmt.Fprintf(mf.buf, "(%s", e.Op())
	}
	for i := 0; i < e.ChildCount(); i++ {
		child := e.Child(i)
		if opt.IsListItemOp(child) {
			child = child.Child(0)
		}
		fmt.Fprintf(mf.buf, " G%d", mf.group(child)+1)
	}
	mf.formatPrivate(e, &physical.Required{})
	mf.buf.WriteString(")")
}

func (mf *memoFormatter) formatBest(best memo.RelExpr, required *physical.Required) {
	fmt.Fprintf(mf.buf, "(%s", best.Op())

	for i := 0; i < best.ChildCount(); i++ {
		fmt.Fprintf(mf.buf, " G%d", mf.group(best.Child(i))+1)

		// Print properties required of the child if they are interesting.
		childReq := BuildChildPhysicalProps(mf.o.mem, best, i, required)
		if childReq.Defined() {
			fmt.Fprintf(mf.buf, "=\"%s\"", childReq)
		}
	}

	mf.formatPrivate(best, required)
	mf.buf.WriteString(")")
}

func (mf *memoFormatter) formatPrivate(e opt.Expr, physProps *physical.Required) {
	private := e.Private()
	if private == nil {
		return
	}

	// Remap special-case privates.
	switch t := e.(type) {
	case *memo.CastExpr:
		private = t.Typ.SQLString()
	}

	// Start by using private expression formatting.
	m := mf.o.mem
	nf := memo.MakeExprFmtCtxBuffer(
		mf.o.ctx, mf.buf, memo.ExprFmtHideAll, mf.redactableValues, m, nil, /* catalog */
	)
	memo.FormatPrivate(&nf, private, physProps)

	// Now append additional information that's useful in the memo case.
	switch t := e.(type) {
	case *memo.ScanExpr:
		tab := m.Metadata().Table(t.Table)
		if tab.ColumnCount() != t.Cols.Len() {
			fmt.Fprintf(mf.buf, ",cols=%s", t.Cols)
		}
		if t.Constraint != nil {
			fmt.Fprintf(mf.buf, ",constrained")
		}
		if t.InvertedConstraint != nil {
			fmt.Fprintf(mf.buf, ",constrained inverted")
		}
		if t.HardLimit.IsSet() {
			fmt.Fprintf(mf.buf, ",lim=%s", t.HardLimit)
		}

	case *memo.IndexJoinExpr:
		fmt.Fprintf(mf.buf, ",cols=%s", t.Cols)

	case *memo.LookupJoinExpr:
		if len(t.KeyCols) > 0 {
			fmt.Fprintf(mf.buf, ",keyCols=%v,outCols=%s", t.KeyCols, t.Cols)
		} else {
			fmt.Fprintf(mf.buf, ",outCols=%s", t.Cols)
		}

	case *memo.ExplainExpr:
		propsStr := t.Props.String()
		if propsStr != "" {
			fmt.Fprintf(mf.buf, " %s", propsStr)
		}

	case *memo.ProjectExpr:
		t.Passthrough.ForEach(func(i opt.ColumnID) {
			fmt.Fprintf(mf.buf, " %s", m.Metadata().ColumnMeta(i).Alias)
		})
	}
}

func firstExpr(expr opt.Expr) opt.Expr {
	if rel, ok := expr.(memo.RelExpr); ok {
		return rel.FirstExpr()
	}
	return expr
} And here is the function you are asked to complete func (mf *memoFormatter) formatCycles() string
func (mf *memoFormatter) formatCycles() string {
	m := mf.o.mem
	var cd cycle.Detector

	if mf.flags != FmtCycle {
		return ""
	}

	addExpr := func(from cycle.Vertex, e opt.Expr) {
		for i := 0; i < e.ChildCount(); i++ {
			child := e.Child(i)
			if opt.IsListItemOp(child) {
				child = child.Child(0)
			}
			to := cycle.Vertex(mf.group(child))
			cd.AddEdge(from, to)
		}
	}

	addGroup := func(from cycle.Vertex, first memo.RelExpr) {
		for member := first; member != nil; member = member.NextExpr() {
			addExpr(from, member)
		}
	}

	// Add an edge to the cycle detector from a group to each of the groups it
	// references.
	for i, e := range mf.groups {
		from := cycle.Vertex(i)
		rel, ok := e.first.(memo.RelExpr)
		if !ok {
			addExpr(from, e.first)
			continue
		}
		addGroup(from, rel)
	}

	// Search for a cycle from the root expression group.
	root := cycle.Vertex(mf.group(m.RootExpr()))
	if <ATTN>cyclePath</ATTN>, ok := cd.FindCycleStartingAtVertex(root); ok {
		mf.buf.Reset()
		mf.buf.WriteString(", cycle=[")
		for i, group := range <ATTN>cyclePath</ATTN> { <MID> 		}
		mf.buf.WriteString("]")
		return mf.buf.String()
	}
	return ""
} Ensure that only missing codes marked as <MID> are returned<|im_end|>
<|im_start|>assistant
			if i > 0 {
				mf.buf.WriteString("->")
			}
			fmt.Fprintf(mf.buf, "G%d", group+1)<|im_end|>
```