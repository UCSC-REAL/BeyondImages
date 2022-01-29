

div="Total-Variation KL"

mi_type="plain"
for DIVI in $div;
do
    for MITYPE in $mi_type;
    do
        echo "running $DIVI, $MITYPE"

        CUDA_VISIBLE_DEVICES=-1 python3  runner.py --div $DIVI --mi_type $MITYPE --e0 0.4 --e1 0.2

        wait
        echo "running $DIVI, $MITYPE -----Done"
	done
    
done