n=15;#the variable that I want to be incremented
next_n=$[$n+1]
sed -i "/#the variable that I want to be incremented$/s/=.*#/=$next_n;#/" ${0}
echo $n

mkdir results/$n

python3 baseline-DDI.py "../../../data/Test-DDI/DrugBank" > results/$n/task9.1.txt
java -jar ../../../eval/evaluateDDI.jar ../../../data/Test-DDI/DrugBank results/$n/task9.1.txt

echo $n >> results/summary.txt
tail -5 results/$n/task9.1_All_scores.log >> results/summary.txt
