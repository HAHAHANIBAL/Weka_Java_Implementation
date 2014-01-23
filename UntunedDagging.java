import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.GridSearch;
import weka.classifiers.mi.CitationKNN;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.BFTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.NBTree;
import weka.core.Instances;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.FileReader;


public class UntunedDagging {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid2_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/hypothyroid2_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        CVParameterSelection ps = new CVParameterSelection();
        ps.setClassifier(new Dagging());
        //find optimal parameter
        //ps.addCVParameter("F 1 5 5");
        //ps.addCVParameter("S 1 10 10");

        Dagging cls = new Dagging();
        //change the base classifier
        //cls.setClassifier(new NBTree());
        //change the parameter for dagging
        //cls.setNumFolds(4);
        //cls.setSeed(6);

        cls.buildClassifier(train);
        //ps.buildClassifier(train);

        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));

        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone2/hypothyroid20.model", cls);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(cls,test);
        Double error_c=eval.errorRate();
        System.out.println(error_c);





    }
}