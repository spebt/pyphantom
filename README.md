# Digital Phantom Generation
This is a `python`-`numpy`-`scipy`-based digital phantom generation codebase.
## Dependency
1. `Python 3`, tested on `Python 3.10.12`
2. Install the python packages with

    ```python3 -m pip install -r requirements.txt```
## Running the code
```
python3 phantom-gen.py [phantom type]
```
* [phantom type] is a required option. Accepted options are:
    1. _hotrod_
    2. _Derenzo_
    3. _derenzo_
    
    The above 3 are treated the same.

    4. _contrast_

    option 4 is work in progress (WIP), it is implemented in the jupyter notebook. But not in the python script yet.

    5. _dot_
### Example command:
```
python3 phantom-gen.py hotrod
```
