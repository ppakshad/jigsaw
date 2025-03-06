import soot.*;
import soot.jimple.*;
import soot.jimple.internal.JAssignStmt;
import soot.tagkit.Tag;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.ExceptionalBlockGraph;
import soot.toolkits.graph.ExceptionalUnitGraph;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static java.lang.System.exit;


public class URL_extractor implements Extractor{

    private Soot_utlilty utility = new Soot_utlilty();
    private CallGraphUtility cgUtil = new CallGraphUtility();

    public ArrayList<SootClass> extract_class_url(String feature) {

        ArrayList<SootClass> to_ret = new ArrayList<>();
        for (SootClass c : Scene.v().getApplicationClasses()) {
            if (utility.isExcludeClass(c)) {
                continue;
            }
            for(SootField f : c.getFields()){
                for(Tag t  :f.getTags()){
                    if(t.toString().contains(feature)){
                        to_ret.add(c);
                    }
                }

            }
            for (SootMethod m : c.getMethods()) {
                if (!m.hasActiveBody()) {
                    continue;
                }
                List<ValueBox> useBoxes = m.getActiveBody().getUseAndDefBoxes();
                for (ValueBox valueBox : useBoxes) {
                    String content = valueBox.getValue().toString();
                    if (content.contains(feature)) {
                        to_ret.add(c);
                    }
                }
            }
        }
        return to_ret;
    }

    private SootMethod find_method_to_extract(String feature,SootClass starting_class,SootMethod starting_method){
        SootMethod to_ret = null;
        boolean done = false;
        Body b = starting_method.getActiveBody();
        ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
        ExceptionalBlockGraph BFG = new ExceptionalBlockGraph(CFG);
        for (Iterator<Block> iter = BFG.iterator(); iter.hasNext(); ) {
            Block block = iter.next();
            if (block.toString().contains(feature)) {
                for(Iterator<Unit> iter_block = block.iterator(); iter_block.hasNext();){
                    Unit u_tmp = iter_block.next();
                    if (u_tmp.toString().contains(feature)){
                        if(u_tmp instanceof InvokeStmt){
                            InvokeExpr invocation = ((InvokeStmt) u_tmp).getInvokeExpr();
                            to_ret = invocation.getMethod();
                            done = true;
                            break;
                        }
                        if (u_tmp instanceof JAssignStmt){
                            for(ValueBox vv : u_tmp.getUseAndDefBoxes()){
                                if(vv.getValue() instanceof InvokeExpr){
                                    to_ret = ((InvokeExpr) vv.getValue()).getMethod();
                                    done = true;
                                    break;
                                }
                            }
                        }
                    }

                }
            }
            if(done){
                break;
            }
        }
        return to_ret;
    }

    @Override
    public SootClass extract_feature(String feature, String name_folder) {
        SootClass slice_class = null;
        ArrayList<SootClass> entrypoints = this.extract_class_url(feature);
        if(!entrypoints.isEmpty()) {
            for (Iterator<SootClass> iter = entrypoints.iterator(); iter.hasNext(); ) {
                SootClass entrypoint = iter.next();
                ArrayList<SootMethod> method_entrypoints = utility.find_method_for_feature(entrypoint, feature);
                ArrayList<String> dependencies_tot = new ArrayList<>();
                ArrayList<String> dependencies = utility.extract_activity_dependencies_PDG(new ArrayList<String>(), entrypoint.getName());
                dependencies_tot.addAll(dependencies);
                boolean done = false;
                Iterator<SootMethod> target_iterator = method_entrypoints.iterator();
                while (done == false && target_iterator.hasNext()) {
                    SootMethod method = target_iterator.next();
                    SootMethod corresponding_sootMethod = this.find_method_to_extract(feature,entrypoint,method);
                    //if returns null it means that there is an assignment of the string --> old method
                    if (corresponding_sootMethod != null){
                        ArrayList<My_slice> slices = utility.extract_method_call_method(entrypoint, method, corresponding_sootMethod);
                        if (!slices.isEmpty()) {
                            My_slice simplest = null;
                            if (slices.size() > 1) {
                                simplest = utility.get_simpler_slice(slices);
                            } else {
                                simplest = slices.get(0);
                            }
                            simplest.setName("Slice" + name_folder + "url");
                            simplest.setFeature("tmp");
                            SootClass tmp = simplest.get_soot_class();
                            System.out.println(simplest.toString());
                            utility.WriteFile("class_of_extraction.txt", entrypoint.getName());
                            slice_class = tmp;
                            done = true;
                        } else {
                            Map<SootClass,ArrayList<SootMethod>> result_method = cgUtil.get_callgraph_for_method(Instrumenter.apkPath, Instrumenter.jarsPath,entrypoint,method);
                            SootClass class_of_url_inner = null;
                            if(!result_method.isEmpty()){
                                class_of_url_inner = result_method.keySet().iterator().next();
                                System.out.println("Found calling class "+class_of_url_inner.getName()+", getting now the dependencies of it ...");
                                slices = utility.extract_method_call_method(class_of_url_inner, result_method.get(class_of_url_inner).get(0), method);
                                if(!slices.isEmpty()) {
                                    My_slice simplest = utility.get_simpler_slice(slices);
                                    simplest.setName("Slice" + name_folder+"url");
                                    simplest.setFeature("tmp");
                                    SootClass tmp = simplest.get_soot_class();
                                    System.out.println(simplest.toString());
                                    utility.WriteFile("class_of_extraction.txt", class_of_url_inner.getName());
                                    slice_class = tmp;
                                    done = true;
                                }else{
                                    System.out.println("No possible slice found :(");
                                }
                        }
                        }
                    }else{
                        Map<SootClass,ArrayList<SootMethod>> result_method = cgUtil.get_callgraph_for_method(Instrumenter.apkPath, Instrumenter.jarsPath,entrypoint,method);
                        SootClass class_of_url_inner = null;
                        if(!result_method.isEmpty()){
                            class_of_url_inner = result_method.keySet().iterator().next();
                            System.out.println("Found calling class "+class_of_url_inner.getName()+", getting now the dependencies of it ...");
                            ArrayList<My_slice> slices = utility.extract_method_call_method(class_of_url_inner,result_method.get(class_of_url_inner).get(0),method);
                            if(!slices.isEmpty()) {
                                My_slice simplest = null;
                                if (slices.size() > 1) {
                                    simplest = utility.get_simpler_slice(slices);
                                } else {
                                    simplest = slices.get(0);
                                }
                                simplest.setName("Slice" + name_folder+"url");
                                simplest.setFeature("tmp");
                                SootClass tmp = simplest.get_soot_class();
                                System.out.println(simplest.toString());
                                utility.WriteFile("class_of_extraction.txt", class_of_url_inner.getName());
                                slice_class = tmp;
                                done = true;
                            }else{
                                System.out.println("No possible slice found :(");
                            }
                        }
                    }

                }
                if (slice_class !=null){
                    utility.extract_classes(dependencies_tot);
                    break;
                }
            }
        }else{
            System.out.println("Sorry the feature has failed to being identified in this apk ...\nExiting\n");
            exit(0);
        }

        return slice_class;
    }
}

