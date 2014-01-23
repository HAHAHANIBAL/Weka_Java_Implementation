import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;



public class VotingWithFilter {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/spambase_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/spambase_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        //set up filter of the data
        AttributeSelection filter = new AttributeSelection();
        BestFirst search = new BestFirst();
        search.setOptions(weka.core.Utils.splitOptions("-D 2 -N 10"));
        filter.setEvaluator(new CfsSubsetEval());
        filter.setSearch(search);
        filter.setInputFormat(train);
        Instances newtrain = Filter.useFilter(train, filter);
        Instances newtest = Filter.useFilter(test, filter);
        newtrain.setClassIndex(newtrain.numAttributes() - 1);
        newtest.setClassIndex(newtest.numAttributes()-1);

        //Classifier[] ClassifierArray=new Classifier[3];
        //ClassifierArray[1]=new J48();
        //ClassifierArray[0]=new NaiveBayes();
        //ClassifierArray[2]=new LogitBoost();
        Vote vs=new Vote();
        //String[] options=new String[3];
        //options[2]="-R MAJ";
        //options[1]="-B weka.classifiers.functions.SMO -B weka.classifiers.bayes.NaiveBayes";
        //options[0]="-S <2>";
        //vs.setOptions(options);
        //ps.setOptions(weka.core.Utils.splitOptions("-P \"I 1.0 10.0 1.0\" -P \"P 1.0 100.0 10.0\" -W \"weka.classifiers.meta.AdaBoostM1\" -- -P 100 -S 1 -I 10 -W \"weka.classifiers.functions.SMO\" -- -V -1 -C 1 -P 1.0E-12"));
        vs.setOptions(weka.core.Utils.splitOptions("-B \"weka.classifiers.functions.SMO\" -B \"weka.classifiers.trees.LMT\" -B \"weka.classifiers.trees.RandomForest -I 10 -K 0\" -R AVG"));

        //String[] options=ps.getBestClassifierOptions();

        //vs.setOptions(options);
        System.out.println(Utils.joinOptions(vs.getOptions()));
        //vs.setClassifiers(ClassifierArray);

        vs.buildClassifier(newtrain);
        //find optimal parameter
        //CVParameterSelection ps= new CVParameterSelection();
        //change the base classifier
        //cls.setClassifier(new NBTree());
        //change the parameter for dagging
        //cls.setNumFolds(1);
        //cls.setSeed(7);
        //cls.buildClassifier(train);
        //System.out.println(vs.getCombinationRule());
        //System.out.println(vs.getOptions());
        PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone5/spambase-L5.txt"));

        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        for (int i = 0; i < newtest.numInstances(); i++) {
            double pred = vs.classifyInstance(newtest.instance(i));
            pw.println(pred);
        }
        pw.close();
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone3/ionosphere.model", vs);





    }
}