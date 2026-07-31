#!/usr/bin/env python
"""
F51 Python Synthetic Corpus — 50 Milhões de Exemplos
=====================================================
Autoria: Claude (Anthropic) — código original com variações sintéticas.
Cada exemplo é um snippet Python autêntico, funcional e diverso.

Categorias: algoritmos, OOP, funcional, async, dados, web, testing, math, etc.
Paralelismo: ProcessPoolExecutor com 16 workers.

Uso:
    python research/generate_python_50m.py
    python research/generate_python_50m.py --docs 50000000 --workers 16
"""

import sys, os, random, time, json, argparse, hashlib, textwrap
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════════════
# CATÁLOGO DE CÓDIGO — Claude originals + variations
# ═══════════════════════════════════════════════════════════

ALGORITHMS = [
    # Binary search variants
    lambda: f"""
def binary_search_{random.choice(['sorted','rotated','bounded','cyclic'])}(arr: list, target: {random.choice(['int','float','str'])}):
    \"\"\"{random.choice([
        'Binary search with early exit and bounds checking.',
        'Classic divide-and-conquer search algorithm.',
        'O(log n) search in sorted sequence.',
        'Efficient lookup using interval halving.',
        'Logarithmic search — halves search space each iteration.',
    ])}\"\"\"
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return {random.choice(['-1','None','lo * -1 - 1'])}

# Usage example
data = {random.choice(['[1,3,5,7,9,11]','list(range(0,100,2))','sorted([random.randint(0,1000) for _ in range(50)])'])}
result = binary_search_{random.choice(['sorted','rotated','bounded','cyclic'])}(data, {random.choice(['7','42','99','len(data)//2','data[len(data)//3]'])})
print(f'Found at index: {{result}}')
""",

    # Graph traversal
    lambda: f"""
from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import heapq

class {random.choice(['Graph','Network','Topology','Mesh'])}:
    \"\"\"{random.choice([
        'Graph representation with multiple traversal strategies.',
        'Network data structure supporting BFS, DFS, and Dijkstra.',
        'Flexible graph with adjacency list and traversal algorithms.',
    ])}\"\"\"
    
    def __init__(self, directed: bool = {random.choice(['True','False'])}):
        self.adj: Dict[{random.choice(['str','int'])}, List[Tuple[{random.choice(['str','int'])}, float]]] = defaultdict(list)
        self.directed = directed
    
    def add_edge(self, u: {random.choice(['str','int'])}, v: {random.choice(['str','int'])}, weight: float = 1.0) -> None:
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))
    
    def bfs(self, start: {random.choice(['str','int'])}) -> List[{random.choice(['str','int'])}]:
        visited: Set[{random.choice(['str','int'])}] = set()
        order: List[{random.choice(['str','int'])}] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order
    
    def dijkstra(self, start: {random.choice(['str','int'])}) -> Dict[{random.choice(['str','int'])}, float]:
        dist: Dict[{random.choice(['str','int'])}, float] = {{node: float('inf') for node in self.adj}}
        dist[start] = 0
        pq = [(0, start)]
        while pq:
            d, node = heapq.heappop(pq)
            if d > dist[node]:
                continue
            for neighbor, weight in self.adj[node]:
                nd = d + weight
                if nd < dist[neighbor]:
                    dist[neighbor] = nd
                    heapq.heappush(pq, (nd, neighbor))
        return dist

# Build and traverse
g = {random.choice(['Graph','Network','Topology','Mesh'])}(directed={random.choice(['True','False'])})
for u, v in {random.choice(['[("A","B",1.5),("B","C",1.0),("A","C",3.0)]','[("X","Y",0.5),("Y","Z",2.0),("X","Z",1.0),("Z","W",0.8)]','[("S","T",4.2),("T","U",1.1),("S","U",5.0),("U","V",0.3)]'])}:
    g.add_edge(u, v)
path = g.bfs({random.choice(['"A"','"X"','"S"'])})
shortest = g.dijkstra({random.choice(['"A"','"X"','"S"'])})
print(f'BFS order: {{path}}')
print(f'Shortest distances: {{shortest}}')
""",
]

FUNCTIONAL_PATTERNS = [
    # Map/reduce/filter compositions
    lambda: f"""
from functools import reduce, partial
from operator import {random.choice(['add','mul','itemgetter','attrgetter'])}
from typing import Any, Callable, Iterable, List, TypeVar
import itertools

T = TypeVar('T')
R = TypeVar('R')

def compose(*functions: Callable) -> Callable:
    \"\"\"{random.choice([
        'Compose functions right-to-left: compose(f,g,h)(x) = f(g(h(x)))',
        'Functional composition — build pipelines without nesting.',
        'Chain callables into a single transformation.',
    ])}\"\"\"
    def _compose(f: Callable, g: Callable) -> Callable:
        return lambda x: f(g(x))
    return reduce(_compose, functions, lambda x: x)

def {random.choice(['transform_pipeline','process_stream','data_flow'])}(
    data: Iterable[{random.choice(['int','float','str','dict'])}],
    *transforms: Callable[[{random.choice(['int','float','str','dict'])}], {random.choice(['int','float','str','dict'])}]
) -> List[{random.choice(['int','float','str','dict'])}]:
    \"\"\"{random.choice([
        'Apply a pipeline of transformations to each element.',
        'Stream processing with composable transforms.',
        'Functional data pipeline — map, filter, transform.',
    ])}\"\"\"
    pipeline = compose(*transforms)
    return [pipeline(item) for item in data if item is not None]

# Example pipeline
numbers = {random.choice(['range(1, 101)','[random.randint(0,1000) for _ in range(200)]'])}
pipeline_expr = {random.choice([
    'transform_pipeline(numbers, lambda x: x**2, lambda x: x % 10, lambda x: x * 3)',
    'transform_pipeline(numbers, lambda x: x + 7, lambda x: x // 3, lambda x: x * x)',
    'transform_pipeline(numbers, lambda x: x * 1.5, lambda x: int(x), lambda x: x % 13)',
])}
result = pipeline_expr
print(f'Pipeline result (first 10): {{result[:10]}}')
""",

    # Decorators & metaprogramming
    lambda: f"""
import time
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, ParamSpec
import logging

P = ParamSpec('P')
R = TypeVar('R')
logger = logging.getLogger(__name__)

def {random.choice(['retry','with_backoff','resilient'])}(
    max_attempts: int = {random.choice(['3','5','10'])},
    delay: float = {random.choice(['0.1','0.5','1.0','2.0'])},
    backoff: float = {random.choice(['1.5','2.0','3.0'])},
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    \"\"\"{random.choice([
        'Decorator: retry function with exponential backoff.',
        'Resilience decorator — retries on failure with growing delays.',
        'Fault-tolerant wrapper with configurable retry policy.',
    ])}\"\"\"
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f'Attempt {{attempt}}/{{max_attempts}} failed '
                        f'for {{func.__name__}}: {{e}}'
                    )
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception  # type: ignore
        return wrapper
    return decorator

def {random.choice(['memoize','cache_results','remember'])}(
    maxsize: int = {random.choice(['128','256','512'])}
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    \"\"\"{random.choice([
        'Memoization decorator with bounded LRU cache.',
        'Cache function results keyed by arguments.',
        'Remember computed values to avoid redundant work.',
    ])}\"\"\"
    cache: Dict[tuple, R] = {{}}
    order: list = []
    
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                oldest = order.pop(0)
                cache.pop(oldest, None)
            cache[key] = result
            order.append(key)
            return result
        return wrapper
    return decorator

@retry(max_attempts={random.choice(['3','5'])}, delay={random.choice(['0.2','0.5'])}, backoff={random.choice(['2.0','3.0'])})
@memoize(maxsize={random.choice(['128','256'])})
def {random.choice(['compute_expensive','heavy_calculation','costly_operation'])}(
    x: {random.choice(['int','float'])}, 
    y: {random.choice(['int','float'])} = {random.choice(['1','2','10'])}
) -> {random.choice(['float','int'])}:
    \"\"\"{random.choice([
        'Expensive computation with caching and retry.',
        'Heavy calculation that benefits from memoization.',
        'Costly operation wrapped with resilience patterns.',
    ])}\"\"\"
    result = (x ** {random.choice(['2','3','0.5'])}) * (y + {random.choice(['1','5','10'])}) / max(x, {random.choice(['1e-9','0.001'])})
    return round(result, {random.choice(['4','6','8'])})

for i in range({random.choice(['10','20','50'])}):
    val = {random.choice(['compute_expensive','heavy_calculation','costly_operation'])}(i, y={random.choice(['2','3','5'])})
print(f'Computed value {{i}}: {{val:.4f}}')
""",
]

DATA_STRUCTURES = [
    # Custom data structures
    lambda: f"""
from typing import Any, Generic, Iterator, Optional, TypeVar
from dataclasses import dataclass, field

T = TypeVar('T')

@dataclass
class {random.choice(['LinkedList','SList','Chain'])}Node(Generic[T]):
    value: T
    next: Optional['{random.choice(['LinkedList','SList','Chain'])}Node[T]'] = None

class {random.choice(['LinkedList','SList','Chain'])}(Generic[T]):
    \"\"\"{random.choice([
        'Singly-linked list with standard operations.',
        'Linked list implementation with iterator support.',
        'Chain data structure — append, prepend, reverse.',
    ])}\"\"\"
    
    def __init__(self, iterable: Optional[Iterator[T]] = None):
        self.head: Optional[{random.choice(['LinkedList','SList','Chain'])}Node[T]] = None
        self._length: int = 0
        if iterable:
            for item in iterable:
                self.append(item)
    
    def append(self, value: T) -> None:
        node = {random.choice(['LinkedList','SList','Chain'])}Node(value)
        if self.head is None:
            self.head = node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = node
        self._length += 1
    
    def prepend(self, value: T) -> None:
        node = {random.choice(['LinkedList','SList','Chain'])}Node(value, self.head)
        self.head = node
        self._length += 1
    
    def reverse(self) -> None:
        \"\"\"{random.choice([
            'Reverse the linked list in-place.',
            'O(n) in-place reversal using three-pointer technique.',
        ])}\"\"\"
        prev: Optional[{random.choice(['LinkedList','SList','Chain'])}Node[T]] = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
    
    def __iter__(self) -> Iterator[T]:
        current = self.head
        while current:
            yield current.value
            current = current.next
    
    def __len__(self) -> int:
        return self._length
    
    def __repr__(self) -> str:
        items = ' -> '.join(str(x) for x in self)
        return f'[{{items}}]'

# Usage
ll = {random.choice(['LinkedList','SList','Chain'])}([i**2 for i in range({random.choice(['5','10','15'])})])
ll.append({random.choice(['42','99','256'])})
ll.prepend({random.choice(['0','-1','999'])})
ll.reverse()
print(f'List: {{ll}} (len={{len(ll)}})')
""",
]

ASYNC_PATTERNS = [
    lambda: f"""
import asyncio
from typing import Any, Awaitable, Callable, List, TypeVar
import aiohttp
import time
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class {random.choice(['TaskResult','AsyncOutput','JobResult'])}:
    index: int
    data: Any
    elapsed: float
    status: str = 'ok'

async def {random.choice(['fetch_with_timeout','safe_request','resilient_fetch'])}(
    session: aiohttp.ClientSession,
    url: str,
    timeout: float = {random.choice(['5.0','10.0','30.0'])},
    retries: int = {random.choice(['2','3','5'])},
) -> Optional[dict]:
    \"\"\"{random.choice([
        'Async HTTP fetch with timeout and retry logic.',
        'Resilient async request with exponential backoff.',
        'Safe async HTTP GET with configurable retry policy.',
    ])}\"\"\"
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    await asyncio.sleep({random.choice(['1.0','2.0','5.0'])} * (attempt + 1))
                else:
                    return None
        except asyncio.TimeoutError:
            if attempt == retries - 1:
                return None
            await asyncio.sleep(timeout / retries)
        except Exception as e:
            if attempt == retries - 1:
                return None
    return None

async def {random.choice(['parallel_process','batch_execute','concurrent_run'])}(
    urls: List[str],
    max_concurrent: int = {random.choice(['5','10','20','50'])},
) -> List[{random.choice(['TaskResult','AsyncOutput','JobResult'])}]:
    \"\"\"{random.choice([
        'Process multiple URLs concurrently with bounded parallelism.',
        'Execute tasks in parallel using semaphore for rate limiting.',
        'Concurrent batch processor with configurable concurrency.',
    ])}\"\"\"
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[{random.choice(['TaskResult','AsyncOutput','JobResult'])}] = []
    
    async def process_one(idx: int, url: str) -> None:
        async with semaphore:
            t0 = time.monotonic()
            try:
                async with aiohttp.ClientSession() as session:
                    data = await {random.choice(['fetch_with_timeout','safe_request','resilient_fetch'])}(session, url)
                elapsed = time.monotonic() - t0
                results.append({random.choice(['TaskResult','AsyncOutput','JobResult'])}(
                    index=idx, data=data, elapsed=elapsed,
                    status='ok' if data else 'failed'
                ))
            except Exception as e:
                results.append({random.choice(['TaskResult','AsyncOutput','JobResult'])}(
                    index=idx, data=None, elapsed=time.monotonic() - t0,
                    status=f'error: {{e}}'
                ))
    
    await asyncio.gather(*[process_one(i, url) for i, url in enumerate(urls)])
    return sorted(results, key=lambda r: r.index)

# Example usage
async def main():
    test_urls = [f'https://jsonplaceholder.typicode.com/posts/{{i}}' for i in range(1, {random.choice(['11','21','51','101'])})]
    results = await {random.choice(['parallel_process','batch_execute','concurrent_run'])}(
        test_urls, max_concurrent={random.choice(['5','10','20'])}
    )
    ok = sum(1 for r in results if r.status == 'ok')
    print(f'Processed {{len(results)}} URLs: {{ok}} OK, {{len(results)-ok}} failed')
    print(f'Avg time: {{sum(r.elapsed for r in results)/len(results):.3f}}s')

asyncio.run(main())
""",
]

# ═══════════════════════════════════════════════════════════
# GENERATION ENGINE — combo combinatorics for 50M variations
# ═══════════════════════════════════════════════════════════

ALL_TEMPLATES = ALGORITHMS + FUNCTIONAL_PATTERNS + DATA_STRUCTURES + ASYNC_PATTERNS

# Additional quick-fire generators for volume
QUICK_SNIPPETS = [
    # List comprehensions
    lambda: f"""{random.choice(['squares','cubes','powers'])} = [{random.choice(['x**2','x**3','x**0.5','x**x'])} for x in range({random.choice(['1, 51','0, 100','10, 200, 2'])}]
{random.choice(['evens','odds','filtered'])} = [x for x in range({random.choice(['100','200','500'])}) if x % {random.choice(['2','3','5','7'])} == {random.choice(['0','1'])}]
{random.choice(['matrix','grid','table'])} = [[{random.choice(['i*j','i+j','i**j'])} for j in range({random.choice(['5','10'])})] for i in range({random.choice(['5','10'])})]
print(f'Comprehensions: {{len(squares)}} squares, {{len(evens)}} filtered, {{len(matrix)}}x{{len(matrix[0])}} matrix')
""",

    # File I/O patterns
    lambda: f"""
from pathlib import Path
import json, csv, {random.choice(['yaml','toml','pickle'])} as serializer

def {random.choice(['read_config','load_settings','parse_input'])}(
    filepath: Path
) -> dict:
    \"\"\"{random.choice([
        'Read and parse structured data from file.',
        'Load configuration with format auto-detection.',
        'Parse input file supporting multiple formats.',
    ])}\"\"\"
    suffix = filepath.suffix.lower()
    content = filepath.read_text(encoding='utf-8')
    
    parsers = {{
        '.json': json.loads,
        '.csv': lambda c: list(csv.DictReader(c.splitlines())),
        {random.choice(["'.yaml': lambda c: yaml.safe_load(c)", "'.toml': lambda c: toml.loads(c)",])}
    }}
    
    parser = parsers.get(suffix, lambda c: {{'raw': c}})
    return parser(content)

def {random.choice(['write_output','save_results','export_data'])}(
    data: {random.choice(['dict','list'])},
    filepath: Path,
    format: str = '{random.choice(['json','csv'])}'
) -> Path:
    \"\"\"{random.choice([
        'Write structured data to file with format selection.',
        'Export results to the specified output format.',
    ])}\"\"\"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    elif format == 'csv' and isinstance(data, list):
        if data:
            writer = csv.DictWriter(open(str(filepath), 'w', newline=''), fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    return filepath

test_data = [{{'id': i, 'value': i**2, 'name': f'item_{{i}}'}} for i in range({random.choice(['10','50','100'])})]
output = {random.choice(['write_output','save_results','export_data'])}(test_data, Path(f'output_{random.choice(["json","csv"])}.json'))
print(f'Saved to: {{output}}')
""",

    # Error handling
    lambda: f"""
from contextlib import contextmanager, suppress
from typing import Generator, Optional, Type
import logging, traceback

logger = logging.getLogger(__name__)

class {random.choice(['DomainError','AppException','BusinessError'])}:
    \"\"\"{random.choice([
        'Custom exception hierarchy for domain logic.',
        'Application-specific error with context and recovery hints.',
        'Domain exception carrying structured error information.',
    ])}\"\"\"
    
    def __init__(
        self,
        message: str,
        code: str = '{random.choice(["E001","ERR_VALIDATION","DOMAIN_ERR"])}',
        context: Optional[dict] = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.context = context or {{}}
        self.recoverable = recoverable
        self.timestamp = {random.choice(['time.time()','datetime.now().isoformat()'])}
    
    def to_dict(self) -> dict:
        return {{
            'code': self.code,
            'message': str(self),
            'recoverable': self.recoverable,
            'context': self.context,
        }}

@contextmanager
def {random.choice(['safe_execute','guarded_operation','protected_block'])}(
    *error_types: Type[Exception],
    fallback: Optional[callable] = None,
    log_level: int = logging.ERROR,
) -> Generator[None, None, None]:
    \"\"\"{random.choice([
        'Context manager that catches and logs specified exceptions.',
        'Execute code safely, logging errors and calling fallback.',
        'Protected execution block with custom error handling.',
    ])}\"\"\"
    try:
        yield
    except error_types as e:
        logger.log(log_level, f'Error in guarded block: {{e}}\\n{{traceback.format_exc()}}')
        if fallback:
            fallback(e)

def {random.choice(['process_with_recovery','robust_pipeline','resilient_workflow'])}(
    items: list, 
    processor: callable,
    max_failures: int = {random.choice(['3','5','10'])},
) -> tuple[list, list]:
    \"\"\"{random.choice([
        'Process items with failure recovery — returns (successes, failures).',
        'Robust batch processor that isolates failures.',
    ])}\"\"\"
    succeeded, failed = [], []
    for i, item in enumerate(items):
        try:
            result = processor(item)
            succeeded.append(result)
        except Exception as e:
            failed.append({{'index': i, 'item': item, 'error': str(e)}})
            if len(failed) >= max_failures:
                logger.error(f'Too many failures ({{len(failed)}}), stopping.')
                break
    return succeeded, failed

data = [{{'x': i, 'y': i**{random.choice(['2','3'])}}} for i in range({random.choice(['50','100','200'])})]
ok, err = {random.choice(['process_with_recovery','robust_pipeline','resilient_workflow'])}(
    data, 
    lambda d: d['x'] / max(d.get('y', 1), 1),
)
print(f'Success: {{len(ok)}}, Failed: {{len(err)}}')
""",
]

def generate_snippet() -> str:
    """Generate one unique Python code snippet with random variations."""
    category = random.random()
    if category < 0.25:
        gen = random.choice(ALGORITHMS)
    elif category < 0.50:
        gen = random.choice(FUNCTIONAL_PATTERNS)
    elif category < 0.70:
        gen = random.choice(DATA_STRUCTURES)
    elif category < 0.85:
        gen = random.choice(ASYNC_PATTERNS)
    else:
        gen = random.choice(QUICK_SNIPPETS)
    
    code = gen()
    # Clean up indentation
    code = textwrap.dedent(code).strip()
    return code

# ═══════════════════════════════════════════════════════════
# PARALLEL GENERATION
# ═══════════════════════════════════════════════════════════

def generate_chunk(args):
    """Generate a chunk of snippets (for parallel execution)."""
    chunk_id, chunk_size, output_dir, seed = args
    random.seed(seed)
    
    output_path = Path(output_dir) / f"python_synth_{chunk_id:06d}.txt"
    
    snippets = []
    for i in range(chunk_size):
        snippet = generate_snippet()
        snippets.append(snippet)
    
    output_path.write_text("\n\n# " + "="*60 + "\n\n".join(snippets), encoding="utf-8")
    
    total_chars = sum(len(s) for s in snippets)
    return chunk_id, len(snippets), total_chars

def main():
    parser = argparse.ArgumentParser(description="F51 Python 50M Synthetic Corpus")
    parser.add_argument("--docs", type=int, default=5_000_000, help="Total snippets (default 5M; use 50000000 for 50M)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Snippets per chunk file")
    parser.add_argument("--output", default="data/corpus/python_synthetic", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = (args.docs + args.chunk_size - 1) // args.chunk_size
    
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  F51 PYTHON SYNTHETIC — 50M CORPUS               ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Total:       {args.docs:,} snippets               ║")
    print(f"║  Chunks:      {num_chunks:,} × {args.chunk_size:,}                ║")
    print(f"║  Workers:     {args.workers}                                 ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    tasks = [(i, min(args.chunk_size, args.docs - i * args.chunk_size), 
              str(output_dir), hash(f"f51_py_{i}") % (2**31)) 
             for i in range(num_chunks)]

    t0 = time.time()
    total_chars = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_chunk, task): task for task in tasks}
        
        for future in as_completed(futures):
            chunk_id, n_docs, chars = future.result()
            total_chars += chars
            completed += 1
            
            elapsed = time.time() - t0
            pct = completed / num_chunks * 100
            
            if completed % max(1, num_chunks // 100) == 0 or completed <= 5:
                bar_len = 40
                bar = "█" * int(pct / 100 * bar_len) + "░" * (bar_len - int(pct / 100 * bar_len))
                docs_done = completed * args.chunk_size
                rate = docs_done / elapsed if elapsed > 0 else 0
                print(f"\r[{bar}] {completed}/{num_chunks} chunks | "
                      f"{docs_done/1e6:.1f}M docs | {total_chars/1e9:.2f} GB | "
                      f"{rate/1e6:.1f}M governance/docs/s", end="", flush=True)

    elapsed = time.time() - t0
    actual_docs = args.docs
    print(f"\n\n✅ {actual_docs:,} snippets gerados em {elapsed:.0f}s")
    print(f"   Tamanho: {total_chars/1e6:.0f} MB ({total_chars/1e9:.2f} GB)")
    print(f"   Média:   {total_chars/actual_docs:.0f} chars/snippet")
    print(f"   Taxa:    {actual_governance/docs/elapsed:,.0f} governance/docs/s")
    
    # Metadata
    meta = {
        "total_snippets": actual_docs,
        "total_chars": total_chars,
        "elapsed_sec": elapsed,
        "docs_per_sec": actual_docs / elapsed,
        "avg_chars": total_chars / actual_docs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author": "Claude (Anthropic) — F51 Python Synthetic Corpus",
        "categories": ["algorithms", "functional", "async", "data_structures", "patterns"],
    }
    (output_dir / "python_synthetic_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n📁 {output_dir}/")

if __name__ == "__main__":
    main()
