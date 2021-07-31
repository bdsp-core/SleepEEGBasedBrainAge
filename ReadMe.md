## Steps
Output: mastersheet.xlsx
1\. $PYTHON step1\_generate\_mastersheet.py

Output: in features folder
2\. $PYTHON step2\_generate\_features.py
3\. $PYTHON step3\_generate\_spindle\_features.py

Output: in output_BA folder
4\. $PYTHON step4\_compute\_BA.py \<dataset name\>
5\. $PYTHON step5\_compute\_stable\_BA.py \<dataset name\>

dataset name = ApoE, STAGES, WSC, ...

## Important requirement
* [Luna](http://zzz.bwh.harvard.edu/luna/) >= v0.25.5 (including LunaR and LunaC)
* Tested only in Ubuntu 18.04

## Notes
* Step 4 computes the brain age in Sun, H., Paixao, L.,..., Thomas, R.J., Cash, S.S., Bianchi, M.T. and Westover, M.B., 2019. Brain age from the electroencephalogram of sleep. *Neurobiology of aging*, 74, pp.112-120.
* Step 5 computes an updated brain age that is longitudinally more stable (not published).
