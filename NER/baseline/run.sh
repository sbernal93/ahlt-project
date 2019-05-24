n=17;#the variable that I want to be incremented
next_n=$[$n+1]
sed -i "/#the variable that I want to be incremented$/s/=.*#/=$next_n;#/" ${0}
echo $n

mkdir results/$n

#python3 baseline-NER.py "results/$n"
python3 baseline-NER.py "../../../data/Test-NER/MedLine" > results/$n/task9.1.txt
java -jar ../../../eval/evaluateNER.jar ../../../data/Test-NER/MedLine results/$n/task9.1.txt

echo $n >> results/summary.txt
tail -5 results/$n/task9.1_scores.log >> results/summary.txt
#echo $n >> results/summary.txt
#tail -5 results/$n/task9.1_scores.log >> results/summary.txt
