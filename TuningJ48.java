import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.GridSearch;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.FileReader;


public class TuningJ48 {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        CVParameterSelection ps = new CVParameterSelection();
        ps.setClassifier(new J48());
        ps.addCVParameter("C 0.01 0.5 50");
        ps.addCVParameter("M 1 10 10");

        J48 cls = new J48();
        //cls.setConfidenceFactor(0.23f);
        //cls.setMinNumObj(10);
        //cls.setBinarySplits(true);
        //cls.setReducedErrorPruning(true);
        //cls.setUseLaplace(true);
        cls.buildClassifier(train);
        ps.buildClassifier(train);

        System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone2/hypothyroid.model", cls);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(cls,test);
        Double error_c=eval.errorRate();
        System.out.println(error_c);





    }
}