import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.misc.SerializedClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.FileReader;


public class Tuning {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/autos_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone2/autos_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        CVParameterSelection ps = new CVParameterSelection();
        //GridSearch ps2 = new GridSearch();
        //ps2.setClassifier(new JRip());
        //ps.setClassifier(new J48());
        //ps.setClassifier(new JRip());
        ps.setClassifier(new J48());
        //vary these parameters
        //ps.setNumFolds(6);
        //ps.setSeed(1);
        //ps.addCVParameter("F 1 20 20");
        //ps.addCVParameter("S 1 10 10");

        //ps.addCVParameter("F 1 5 5");
        //ps.addCVParameter("N 1 5 5");
        //ps.addCVParameter("O 1 5 5");
        //ps.addCVParameter("S 1 5 5");
        //ps.addCVParameter("C 0.01 0.1 50");
        //ps.addCVParameter("M 1 10 10");

        //ps2.setXMin(0.01);
        //ps2.setXMax(0.5);
        //ps2.setXStep(50);
        //ps2.setYMin(1);
        //ps2.setYMax(10);
        //ps2.setYStep(10);



        ps.buildClassifier(train);
        System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        //System.out.println(ps2.getOptions());
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone2/autos.model", ps);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(ps,test);
        Double error_c=eval.errorRate();
        System.out.println(error_c);





    }
}