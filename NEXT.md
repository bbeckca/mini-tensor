* broadcast shape inference
    - aligns shapes from the right
    - allows dimensions of equal size or size 1
    - throws if broadcasting is invalid
    - produces output shape

* broadcast-aware element access
    - if one side has size 1 in dimension, use % to wrap the index to simulate repeating dimensions

* virtual expand
    - returns a new tensor view (or object) with a broadcasted shape
    - validates compability
    - set new shape metadata
    - shares data (no copy)
    - uses access logic w/ broadcasting support using % based indexing


    