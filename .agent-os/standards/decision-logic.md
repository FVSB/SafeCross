# Decision Logic

Standards for the `safecross/` decision module — the core of SafeCross.

## Public API

The module exposes exactly two public functions:

```python
from safecross.decide import can_cross, decide_verbose

# Simple: returns bool
result: bool = can_cross("path/to/image.jpg", conf=0.75)

# Verbose: returns diagnostic dict
info: dict = decide_verbose("path/to/image.jpg", conf=0.75)
```

Always maintain both. `can_cross()` for production/real-time use;
`decide_verbose()` for debugging, logging, and testing.

## Return Types

**`can_cross()`** — always returns `bool`:
- `True` = safe to cross
- `False` = wait or ask for help

**`decide_verbose()`** — always returns `dict` with these keys:
```python
{
    "can_cross": bool,
    "crosswalk_detected": bool,
    "n_crosswalks": int,
    "light_detected": bool,
    "light_color": str | None,   # e.g., "verde", "rojo"
    "reason": str,               # human-readable explanation
}
```

Never change these key names without updating all callers.

## Decision Order (Sequential Checks)

Always apply checks in this order — exit early on first failure:

1. **Crosswalk present?** → `crosswalks.pt`
   - If 0 crosswalks detected → `return False`
2. **Exactly 1 traffic light?** → `lights.pt`
   - If ≠ 1 light detected → `return False`
3. **Light is green?** → check class name == `"verde"`
   - If not green → `return False`
4. *(Future)* **No vehicles in crosswalk?** → `persons_cars.pt`
   - If large vehicle blocking → `return False`
5. All checks passed → `return True`

This order minimizes model loading — if no crosswalk, skip loading lights model.

## Green Light Class Name

The traffic light model uses Spanish class names. The green class is `"verde"`:

```python
_GREEN_CLASS_NAME = "verde"

if class_names[detected_class_id] != _GREEN_CLASS_NAME:
    return False
```

Do not hardcode `"verde"` inline — use the module-level constant.

## Confidence Threshold

Default is `0.75`. Always accept it as a parameter — never hardcode in callers:

```python
def can_cross(img_path: str, conf: float = DEFAULT_CONF) -> bool:
```

## Error Handling

- Raise `FileNotFoundError` if image path does not exist
- Raise `FileNotFoundError` if a weight file is missing (with helpful message)
- Do not catch YOLO inference errors — let them propagate

## Weights Path Convention

```python
_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEIGHTS_DIR  = os.path.join(_REPO_ROOT, "weigths")   # note: typo is intentional
```

Do not modify the path resolution — all other code depends on this convention.
