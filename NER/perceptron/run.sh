n=9;#the variable that I want to be incremented
next_n=$[$n+1]
sed -i "/#the variable that I want to be incremented$/s/=.*#/=$next_n;#/" ${0}
echo $n

mkdir results/$n

python3 extract-features.py ../../../data/Train/MedLine > results/$n/train.cod
python3 train-perceptron.py results/$n/perceptron.pickle results/$n/vectorizer.pickle < results/$n/train.cod

# encode test corpus into features and apply learned model
python3 extract-features.py ../../../data/Test-NER/MedLine > results/$n/test.cod
python3 predict-perceptron.py results/$n/perceptron.pickle results/$n/vectorizer.pickle < results/$n/test.cod  > results/$n/task9.1.txt

# evaluate results with official scorer
java -jar ../../../eval/evaluateNER.jar ../../../data/Test-NER/MedLine results/$n/task9.1.txt

echo $n >> results/summary.txt
tail -5 results/$n/task9.1_scores.log >> results/summary.txt
