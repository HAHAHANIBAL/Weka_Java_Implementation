import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Dagging;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class Deserialization {
    public static void main(String[] args) throws Exception{
        SerializedClassifier cls = new SerializedClassifier();
        cls.setModelFile(new File("/Weka-3-6/ProjectMilestone2/hypothyroid2.model"));
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid2_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid2_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(cls,test);
        Double error_c=eval.errorRate();


        Classifier cls_2 = new NaiveBayes();
        Instances train_nb = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid2_train.arff")));
        Instances test_nb = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid2_test.arff")));
        train_nb.setClassIndex(train_nb.numAttributes() - 1);
        test_nb.setClassIndex(test_nb.numAttributes()-1);
        cls_2.buildClassifier(train_nb);
        Evaluation eval_nb=new Evaluation(train_nb);
        eval_nb.evaluateModel(cls_2,test_nb);
        Double error_nb=eval_nb.errorRate();
        System.out.println(error_c/error_nb);



    }
}
