#!/bin/bash

#!/bin/bash

seg_id=$1
total_seg=$2

# run the eval script directly in the current job
./eval/SE_eval.sh "$seg_id" "$total_seg"