import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Dagging;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;


public class Weka {

    public static void main(String[] args) throws Exception{

        // train target classifier
        Classifier cls_1 = new J48();
        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/anneal_train.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        cls_1.buildClassifier(train);
        // serialization
        weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone2/anneal.model", cls_1);



    }
}