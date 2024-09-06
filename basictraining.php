<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

echo "Loading data into memory\n";

$training = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$testing = $training->randomize()->take(10);

$estimator = new KNearestNeighbors(5);

echo "Training ... \n";

$estimator->train($training);

echo "Predict ... \n";

$predictions = $estimator->predict($testing);
//$maxKeyLength = max(array_map('strlen', array_keys($predictions)));
$maxKeyLength=20;

echo "\n";
echo 'No|       Label          |   Prediction'."\n";
echo '--+----------------------+----------------'."\n";
foreach ($predictions as $key => $predictValue) {
    //$key = str_pad($key, $maxKeyLength);
    $testingValue = str_pad($testing[$key][4], $maxKeyLength);
    $value = str_pad($predictValue, $maxKeyLength);
    echo $key . ' | ' . $testingValue . ' | ' . $predictValue . "\n";
}
echo "\n";

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy is {$score}\n";