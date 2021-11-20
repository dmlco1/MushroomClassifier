import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Mushroom {
    public static void main(String[] args) throws Exception {
        // Get dataset
        DataSource data = new DataSource("ARFF/mushroom.arff");
        if (data == null) {
            System.err.println("Can't load file");
            System.exit(1);
        }
        Instances dataset = data.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Generated model
        J48 classifier = new J48();
        classifier.buildClassifier(dataset);

        // Visualize decision tree
        Visualizer v = new Visualizer();
        v.start(classifier);

        //cross validation test
        Evaluation eval = new Evaluation(dataset); eval.crossValidateModel(classifier, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n ", false));
        System.out.println(eval.toMatrixString());
        System.out.println(classifier);
    }
}
