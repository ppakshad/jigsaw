import fj.test.Bool;
import soot.*;
import soot.jimple.IfStmt;
import soot.jimple.InvokeExpr;
import soot.jimple.InvokeStmt;
import soot.jimple.internal.JAssignStmt;
import soot.options.Options;
import soot.tagkit.Tag;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.ExceptionalBlockGraph;
import soot.toolkits.graph.ExceptionalUnitGraph;
import soot.toolkits.graph.pdg.HashMutablePDG;
import soot.toolkits.graph.pdg.PDGNode;
import soot.util.Chain;

import java.io.*;
import java.util.*;

public class Soot_utlilty {
    protected static List<String> excludePackagesList = new ArrayList<String>();
    protected static List<String> primitiveList = new ArrayList<String>();


    static
    {
        excludePackagesList.add("java.");
        excludePackagesList.add("android.");
        excludePackagesList.add("javax.");
        excludePackagesList.add("android.support.");
        excludePackagesList.add("junit.");
        excludePackagesList.add("org.w3c");
        excludePackagesList.add("org.xmlpull");
        excludePackagesList.add("org.xml.sax.");
        excludePackagesList.add("org.json");
        excludePackagesList.add("org.apache.http.");
        excludePackagesList.add("com.google.android");
        excludePackagesList.add("com.android.");
    }

    static
    {
        primitiveList.add("java.");
        primitiveList.add("android.");
        primitiveList.add("javax.");
        primitiveList.add("android.support.");
        primitiveList.add("junit.");
        primitiveList.add("org.w3c");
        primitiveList.add("org.xmlpull");
        primitiveList.add("org.xml.sax.");
        primitiveList.add("org.json");
        primitiveList.add("org.apache.http.");
        primitiveList.add("com.google.android");
        primitiveList.add("com.android.");
        primitiveList.add("int");
        primitiveList.add("String");
        primitiveList.add("dalvik.");
        primitiveList.add("byte");
        primitiveList.add("boolean");
        primitiveList.add("short");
        primitiveList.add("long");
        primitiveList.add("char");
        primitiveList.add("void");
        primitiveList.add("double");
        primitiveList.add("float");
        primitiveList.add("null");
    }




//global utility function
    boolean isExcludeClass(SootClass sootClass)
    {
        if (sootClass.isPhantom())
        {
            return true;
        }

        String packageName = sootClass.getPackageName();
        for (String exclude : primitiveList)
        {
            if (packageName.startsWith(exclude) && !packageName.startsWith("android.support."))
            {
                return true;
            }
        }

        return false;
    }

    boolean isExcludeClass(String sootClass){


        for (String exclude : primitiveList)
        {
            if (sootClass.startsWith(exclude) && !sootClass.startsWith("android.support."))

            {
                return true;
            }
        }

        return false;
    }

    public ArrayList<SootClass> extract_class_call(String feature) {
        ArrayList<SootClass> to_ret = new ArrayList<>();
        for (SootClass c : Scene.v().getApplicationClasses()) {
            if (this.isExcludeClass(c)) {
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

    public boolean get_if_ancient (SootClass c, String comparison){
        SootClass tmp= c;
        while(tmp.hasSuperclass()){
            if(!tmp.getName().equals(comparison)) {
                tmp = tmp.getSuperclass();
            }else{
                return true;
            }
        }
        return false;
    }

    public SootMethod find_SootMethod(String method,String class_){
        SootMethod to_ret = null;
            String class_tmp = class_.replace("/", ".");
            for (SootClass c : Scene.v().getApplicationClasses()) {
                for (SootMethod m : c.getMethods()) {
                    if (!m.hasActiveBody()) {
                        continue;
                    }
                    if (m.getSignature().contains(method) && m.getSignature().contains(class_tmp)) {
                        to_ret = m;
                        break;
                    }
                }
            }

        return to_ret;
    }

    public SootMethod find_SootMethodNoClass(String method,SootMethod method_){
        SootMethod to_ret = null;
        boolean found = false;

        ArrayList<String> class_tmp = new ArrayList<>();
        if (!method_.hasActiveBody()) {
            return null;
        }
        Body b = method_.getActiveBody();
        ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
        ExceptionalBlockGraph BFG = new ExceptionalBlockGraph(CFG);
        for (Iterator<Block> iter = BFG.iterator(); iter.hasNext(); ) {
            Block block = iter.next();
            if (block.toString().contains(method)) {
                for(Iterator<Unit> iter_block = block.iterator(); iter_block.hasNext();){
                    Unit u_tmp = iter_block.next();
                    if (u_tmp instanceof InvokeStmt && u_tmp.toString().contains(method)){
                        InvokeExpr invocation = ((InvokeStmt) u_tmp).getInvokeExpr();
                        to_ret = invocation.getMethod();
                        found = true;
                        break;
                    }
                    if (u_tmp instanceof JAssignStmt){
                        for(ValueBox vv : u_tmp.getUseAndDefBoxes()){
                            if(vv.getValue() instanceof InvokeExpr){
                                to_ret = ((InvokeExpr) vv.getValue()).getMethod();
                                found = true;
                                break;
                            }
                        }
                    }

                }
            }
            if (found){
                break;
            }
        }
        return to_ret;
    }


    public void extract_classes(ArrayList<String> dependencies) {
        if(Instrumenter.DEBUG) {
            System.out.println("Dependencies found "+dependencies);
        }
        String out = "";
        for(Iterator<SootClass> iter_main = Scene.v().getApplicationClasses().snapshotIterator(); iter_main.hasNext();){
            SootClass c = iter_main.next();

            if(!dependencies.contains(c.getName()) || c.getName().contains("[]") || c.getName().length() ==1){
                c.setPhantomClass();
            }else{

                c.setApplicationClass();
                out = out+c.getName()+"\n";

            }
        }
        if(out.length()>1) {
            this.WriteFile("classes.txt", out);
        }
    }

    //convert to smali name, for matching in jimple files
    public String to_smali(String feature){
        String tmp =feature.replace(".","/");
        tmp= "L"+tmp;
        return  tmp;
    }

    public String to_underline(String feature){
        String tmp =feature.replace(".","_");
          return  tmp;
    }

    public void clean_final_export (String path){
        File to_hold = new File(path);
        ArrayList<SootClass> to_eliminate =new ArrayList<>();
        ArrayList<SootClass> to_readd =new ArrayList<>();

        for (SootClass sc : Scene.v().getApplicationClasses()){
            to_eliminate.add(sc);
        }
        for(SootClass ss : to_eliminate){
            ss.setPhantomClass();
        }
        try {
            BufferedReader reader =new BufferedReader( new FileReader(to_hold));
            String line = reader.readLine();
            while (line != null){
                for (SootClass sc : Scene.v().getPhantomClasses()){
                    if(sc.getName().equals(line.replace("\n",""))){
                        to_readd.add(sc);
                        break;
                    }
                }
                line = reader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(SootClass ss:to_readd){
            ss.setApplicationClass();
        }
    }

    public ArrayList<SootMethod> find_method_for_feature(SootClass searched, String feature) {
        ArrayList<SootMethod> to_ret = new ArrayList<>();
        String correspondence = "";
        if(searched.getFields().size()>0){
            for(SootField f : searched.getFields()){
                for(Tag t  :f.getTags()){
                    if(t.toString().contains(feature)){
                        correspondence = feature+"|"+f.getName();
                        break;
                    }
                }

            }
        }
        for(SootMethod m : searched.getMethods()){
            if(!m.hasActiveBody()){
                continue;
            }
            Body b = m.getActiveBody();
            if(correspondence.isEmpty()){
                if(b.toString().contains(feature)){
                    to_ret.add(m);
                }
            }else{
                String field = correspondence.split("\\|")[1];
                if(b.toString().contains(field)){
                    to_ret.add(m);
                }
            }
        }
        return to_ret;
    }

    //initial Soot settings
    public void initSoot(String apkPath,String output_folder){

        Options.v().set_src_prec(Options.src_prec_apk);

        Options.v().set_output_format(Instrumenter.output_format);

        Options.v().set_output_dir(output_folder);

        String androidJarPath = Scene.v().getAndroidJarPath(Instrumenter.jarsPath, apkPath);

        Instrumenter.jarsPath = androidJarPath;

        List<String> pathList = new ArrayList<String>();

        pathList.add(apkPath);

        pathList.add(androidJarPath);

        Options.v().set_process_dir(pathList);

        Options.v().set_force_android_jar(apkPath);

        Options.v().set_keep_line_number(true);

        Options.v().set_process_multiple_dex(true);

        Options.v().set_allow_phantom_refs(true);

        Options.v().set_whole_program(true);

        Options.v().set_wrong_staticness(Options.wrong_staticness_fix);

        Options.v().set_exclude(excludePackagesList);

        Options.v().set_no_bodies_for_excluded(true);

        Scene.v().loadNecessaryClasses();

        PackManager.v().runPacks();

    }

    //Slice Utility
    public My_slice get_simpler_slice(ArrayList<My_slice> slices){
        Map<Double,My_slice> tmp = new TreeMap<>();
        for(My_slice sl : slices){
            try {
                tmp.put(sl.slice_complexity_score(), sl);
            }catch(RuntimeException e){
                System.out.println("Not possible to extract correctly this slice, passing next");
            }
        }

        return ((TreeMap<Double, My_slice>) tmp).firstEntry().getValue();
    }

    public ArrayList<String>  create_dependencies_file(My_slice slice){
        System.out.println("Create slice dependency files ");

        ArrayList<String> slice_dependencies = slice.get_dependencies();
        String out = slice.getName()+"\n";
        for(String s : slice_dependencies){
            out+= s+"\n";
        }
        WriteFile("slice_dependencies.txt",out);
        WriteFile("feature.txt",to_smali(slice.getFeature()));
        return slice_dependencies;
    }

    public void WriteFile(String name,String content){
        File file = new File(Instrumenter.output_dir + "/"+name);
        if(!file.exists()){
            File folder = new File(Instrumenter.output_dir);
            folder.mkdirs();
            try {
                file.createNewFile();
                FileWriter fileWriter = new FileWriter(file);
                fileWriter.write(content);
                fileWriter.flush();
                fileWriter.close();
                if(Instrumenter.DEBUG) {
                    System.out.println("DEBUG : New file created : "+file.getAbsolutePath());
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }else{
            try {
                file.createNewFile();
                FileWriter fileWriter = new FileWriter(file);
                fileWriter.append(content);
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public ArrayList<My_slice> extract_method_call_method(SootClass entrypoint, SootMethod method_containing, SootMethod method_searched) {
        Boolean extraction_finished =false;
        ArrayList<My_slice> soot_slice = new ArrayList<>();
        System.out.println("Trying to extract method "+method_searched.getSignature()+" from "+entrypoint.getName()+" contained in method "+method_containing.getSignature());
        for (SootMethod m : entrypoint.getMethods()) {
            if (!m.hasActiveBody()) {
                continue;
            }
            Body b = m.getActiveBody();
            if (b.toString().contains(method_searched.getSubSignature()) && m.getSignature().equals(method_containing.getSignature())) {

                ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
                ExceptionalBlockGraph BFG = new ExceptionalBlockGraph(CFG);
                for (Iterator<Block> iter = BFG.iterator(); iter.hasNext(); ) {
                    Block block = iter.next();
                    if (block.toString().contains(method_searched.getSubSignature())) {
                        ArrayList<Unit> units = new ArrayList<>();
                        boolean after = false;
                        for(Iterator<Unit> iter_block = block.iterator(); iter_block.hasNext();){
                            Unit u_tmp = iter_block.next();
                            if(!(u_tmp instanceof IfStmt) && !after) {
                                units.add(u_tmp);
                            }
                            if (u_tmp instanceof InvokeStmt && u_tmp.toString().contains(method_searched.getSubSignature())){
                                InvokeExpr invocation = ((InvokeStmt) u_tmp).getInvokeExpr();
                                if (invocation.getMethod().equals(method_searched)) {
                                    after = true;
                                    break;
                                }
                            }
                            if (u_tmp instanceof JAssignStmt){
                                for(ValueBox vv : u_tmp.getUseAndDefBoxes()){
                                    if(vv.getValue() instanceof InvokeExpr){
                                        if (((InvokeExpr) vv.getValue()).getMethod().equals(method_searched)) {
                                            after = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            for(ValueBox v : u_tmp.getUseAndDefBoxes()){
                                if(v.getValue().toString().contains(method_searched.getDeclaringClass().getName()) && v.getValue().toString().contains(method_searched.getSignature())){
                                    after=true;
                                    break;
                                }
                            }
                        }
                        boolean completed = false;
                        ArrayList<Local> locals_to_export = new ArrayList<>();
                        while (!completed) {
                            ArrayList<String> locals_to_add = new ArrayList<>();
                            for (Iterator<Unit> iter_un = units.iterator(); iter_un.hasNext(); ) {
                                Unit un = iter_un.next();
                                //TODO changed getDefBoxes in get UseAndDefBoxes
                                List<ValueBox> def = un.getUseAndDefBoxes();
                                for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                                    ValueBox def_ = iter_def.next();
                                    if (!locals_to_add.contains(def_.getValue().toString()) ) {
                                        locals_to_add.add(def_.getValue().toString());
                                    }
                                }
                            }
                            for (Iterator<Local> iter_locals = m.getActiveBody().getLocals().iterator(); iter_locals.hasNext(); ) {
                                Local tmp_local = iter_locals.next();
                                if (locals_to_add.contains(tmp_local.getName()) && !locals_to_export.contains(tmp_local)) {
                                    locals_to_export.add(tmp_local);
                                }
                            }
                            ArrayList<String> missing_names = this.missing_values_units(units, locals_to_export);
                            if (!missing_names.isEmpty()) {
                                ArrayList<Unit> missing_units = new ArrayList<>();
                                for (String missing : missing_names) {
                                    for (Iterator<Unit> iter_un = block.getBody().getUnits().iterator(); iter_un.hasNext(); ) {
                                        Unit un = iter_un.next();
                                        List<ValueBox> def = un.getDefBoxes();
                                        if (!def.isEmpty()) {
                                            for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                                                ValueBox def_ = iter_def.next();
                                                if (def_.getValue().toString().equals(missing)) {
                                                    missing_units.add(un);
                                                }
                                            }
                                        }
                                    }
                                }
                                units.addAll(0, missing_units);
                            }else{
                                completed = true;
                            }

                        }

                        My_slice slice = new My_slice(locals_to_export,units);
                        System.out.println(slice.toString());
                        soot_slice.add(slice);

                    }
                }
            }else if (method_searched.getSignature().contains("<clinit>") &&  m.getSignature().equals(method_containing.getSignature()) && b.toString().contains(method_searched.getDeclaringClass().getName()) ){
                if(Instrumenter.DEBUG) {
                    System.out.println("DEBUG : Clinit case ... ");
                }
                ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
                ExceptionalBlockGraph BFG = new ExceptionalBlockGraph(CFG);
                for (Iterator<Block> iter = BFG.iterator(); iter.hasNext(); ) {
                    Block block = iter.next();
                    if (block.toString().contains(method_searched.getDeclaringClass().getName())) {
                        ArrayList<Unit> units = new ArrayList<>();
                        boolean after =false;
                        for(Iterator<Unit> iter_block = block.iterator(); iter_block.hasNext();){
                            Unit u_tmp = iter_block.next();
                            if(!(u_tmp instanceof IfStmt) && !after) {
                                units.add(u_tmp);
                            }
                            for(ValueBox v : u_tmp.getUseAndDefBoxes()){
                                if(v.getValue().toString().contains(method_searched.getDeclaringClass().getName())){
                                    after=true;
                                    break;
                                }
                            }
                        }
                        boolean completed = false;
                        ArrayList<Local> locals_to_export = new ArrayList<>();
                        while(!completed) {
                            ArrayList<String> locals_to_add = new ArrayList<>();

                            for (Iterator<Unit> iter_un = units.iterator(); iter_un.hasNext(); ) {
                                Unit un = iter_un.next();
                                List<ValueBox> def = un.getUseAndDefBoxes();
                                for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                                    ValueBox def_ = iter_def.next();
                                    if (!locals_to_add.contains(def_.getValue().toString())) {
                                        locals_to_add.add(def_.getValue().toString());
                                    }
                                }

                            }
                            for (Iterator<Local> iter_locals = m.getActiveBody().getLocals().iterator(); iter_locals.hasNext(); ) {
                                Local tmp_local = iter_locals.next();
                                if (locals_to_add.contains(tmp_local.getName())  && !locals_to_export.contains(tmp_local)) {
                                    locals_to_export.add(tmp_local);
                                }
                            }
                            ArrayList<String> missing_names = this.missing_values_units(units, locals_to_export);
                            if (!missing_names.isEmpty()) {
                                ArrayList<Unit> missing_units = new ArrayList<>();
                                for (String missing : missing_names) {
                                    for (Iterator<Unit> iter_un = block.getBody().getUnits().iterator(); iter_un.hasNext(); ) {
                                        Unit un = iter_un.next();
                                        List<ValueBox> def = un.getDefBoxes();
                                        if (!def.isEmpty()) {
                                            for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                                                ValueBox def_ = iter_def.next();
                                                if (def_.getValue().toString().equals(missing)) {
                                                    missing_units.add(un);
                                                }
                                            }
                                        }
                                    }
                                }
                                units.addAll(0, missing_units);
                            }

                        }
                        My_slice slice = new My_slice(locals_to_export,units);
                        System.out.println(slice.toString());
                        soot_slice.add(slice);

                    }
                }
            }
        }
        for(My_slice sl : soot_slice){
            this.add_dependencies(sl);
        }

        if(Instrumenter.DEBUG) {
            System.out.println("DEBUG : writing file  ./slices_classes.txt ");
        }
        return soot_slice;
    }

    public ArrayList<String> missing_values_units(ArrayList<Unit> slice,ArrayList<Local> locals) {
        ArrayList<String> tmp = new ArrayList<>();
        ArrayList<String> tmp_locals = new ArrayList<>();
        for (Iterator<Local> iter_local = locals.iterator();iter_local.hasNext();){
            Local tmp_local = iter_local.next();
            if (tmp_local.getName().startsWith("$")) {
                tmp_locals.add(tmp_local.getName().substring(1));
            }else{
                tmp_locals.add(tmp_local.getName());
            }
        }
        for (Iterator<Unit> iter = slice.iterator(); iter.hasNext(); ) {
            Unit un = iter.next();
            List<ValueBox> values = un.getUseBoxes();
            for(Iterator<ValueBox> iter_use = values.iterator(); iter_use.hasNext();){
                ValueBox use = iter_use.next();
                if((use.getValue().toString().contains("r") || use.getValue().toString().contains("i")) && !tmp_locals.contains(use.getValue().toString()) && use.getValue().toString().length() <= 4 && !tmp.contains(use.getValue().toString())){
                    tmp.add(use.getValue().toString());
                }
            }
        }
        for (Iterator<Unit> iter = slice.iterator(); iter.hasNext(); ) {
            Unit un = iter.next();
            List<ValueBox> def= un.getDefBoxes();
            if(!def.isEmpty()) {
                for (Iterator<ValueBox> iter_def = def.iterator(); iter_def.hasNext(); ) {
                    ValueBox def_ = iter_def.next();
                    if (def_.getValue().toString().startsWith("$") && tmp.contains(def_.getValue().toString())) {
                        tmp.remove(def_.getValue().toString());
                    }
                }
            }
        }
        return tmp;

    }

    public void  add_dependencies(My_slice slice){
        ArrayList<Unit> units = slice.getMy_units();
        ArrayList<Local> locals = slice.getMy_locals();
        String out = "";
        for(Unit u : units){
            for(ValueBox v : u.getUseAndDefBoxes()){
                if(!this.isExcludeClass(v.getValue().getType().toString())) {
                    for(String s : extract_activity_dependencies_PDG(new ArrayList<String>(),v.getValue().getType().toString())){
                        SootClass sc = Scene.v().getSootClass(s);
                        sc.setApplicationClass();
                        out = out +sc.getName()+"\n";
                    }
                }
            }
        }
        for(Local l : locals){
            if(!this.isExcludeClass(l.getType().toString())) {
                for(String s : extract_activity_dependencies_PDG(new ArrayList<String>(),l.getType().toString())){
                    SootClass sc = Scene.v().getSootClass(s);
                    sc.setApplicationClass();
                    out = out +sc.getName()+"\n";
                }
            }
        }
        //WriteFile("classes.txt", out);
    }

    public   ArrayList<String> extract_activity_dependencies_PDG(ArrayList<String> dependencies, String feature){
        ArrayList<String> new_dep = new ArrayList<String>();
        if(!dependencies.contains(feature)){
            dependencies.add(feature);
        }
        for (SootClass c : Scene.v().getApplicationClasses()) {
            if(c.getName().equals(feature)){
                Chain<SootClass> interfaces = c.getInterfaces();
                Chain<SootField> fields = c.getFields();
                List<SootMethod> methods = c.getMethods();

                if(c.hasSuperclass()){
                    SootClass superclass = c.getSuperclass();
                    if (!this.isExcludeClass(superclass)) {
                        new_dep.add(superclass.getName());
                    }
                }

                for(Iterator<SootField> iter = fields.iterator(); iter.hasNext();){
                    SootField field= iter.next();
                    if (!this.isExcludeClass(field.getType().toString()) && !field.getType().toString().contains("[]")) {
                        if (!dependencies.contains(field.getType().toString()) && !new_dep.contains(field.getType().toString())) {
                            new_dep.add(field.getType().toString());
                        }
                    }

                }
                //handle interfaces
                for(Iterator<SootClass> iter = interfaces.iterator(); iter.hasNext();){
                    SootClass field= iter.next();
                    if (!this.isExcludeClass(field.getType().toString())) {
                        if (!dependencies.contains(field.getType().toString()) && !new_dep.contains(field.getType().toString())) {
                            new_dep.add(field.getType().toString());
                        }
                    }

                }

                for (SootMethod m : methods ){
                    if (!m.hasActiveBody()) {
                        continue;
                    }
                    Body b = m.getActiveBody();
                    boolean able =true;
                    try{
                        ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
                        HashMutablePDG PDG = new HashMutablePDG(CFG);
                        for(Iterator <PDGNode> iter = PDG.iterator(); iter.hasNext();){
                            PDGNode node = iter.next();
                            for (PDGNode a : node.getDependents()){
                                Block block = (Block) a.getNode();
                                for (Iterator<Unit> iter_u = block.iterator(); iter_u.hasNext();){
                                    Unit unit = iter_u.next();
                                    for (ValueBox v : unit.getUseAndDefBoxes()){
                                        String tmp_feat =v.getValue().getType().toString();
                                        if(!dependencies.contains(tmp_feat) && !this.isExcludeClass(tmp_feat) && !new_dep.contains(tmp_feat)  && !tmp_feat.contains("[")){
                                            new_dep.add(tmp_feat);
                                        }
                                        String tmp =v.getValue().toString();
                                        if (tmp.startsWith("class")){
                                            String dep = tmp.split("\"")[1].split(";")[0].substring(1).replace("/",".");
                                            if (!this.isExcludeClass(dep) && !dep.contains("[]") && dep!= feature){
                                                if(!dep.equals(c.getName()) ){
                                                    if(!dependencies.contains(dep) && !new_dep.contains(dep) ){
                                                        new_dep.add(dep);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                            }
                        }
                    }catch(Exception e){
                        able = false;
                    }
                    if (able == false){
                        for (Iterator<Local> iter_local = b.getLocals().iterator(); iter_local.hasNext();){
                            Local local_tmp = iter_local.next();
                            if (!this.isExcludeClass(local_tmp.getType().toString()) && !local_tmp.getType().toString().contains("[]")){
                                if(!local_tmp.getType().toString().equals(c.getName()) ){
                                    if(!dependencies.contains(local_tmp.getType().toString()) && !new_dep.contains(local_tmp.getType().toString())){
                                        new_dep.add(local_tmp.getType().toString());
                                    }
                                }
                            }
                        }
                        for (Iterator<ValueBox> iter_units = b.getUseAndDefBoxes().iterator() ; iter_units.hasNext();){
                            ValueBox value = iter_units.next();
                            String tmp_feat =value.getValue().getType().toString();
                            if(!dependencies.contains(tmp_feat) && !this.isExcludeClass(tmp_feat) && !new_dep.contains(tmp_feat)  && !tmp_feat.contains("[")){
                                new_dep.add(tmp_feat);
                            }
                            String tmp =value.getValue().toString();
                            if (tmp.startsWith("class")){
                                String dep = tmp.split("\"")[1].split(";")[0].substring(1).replace("/",".");
                                if (!this.isExcludeClass(dep) && !dep.contains("[]") && dep!= feature){
                                    if(!dep.equals(c.getName()) ){
                                        if(!dependencies.contains(dep) && !new_dep.contains(dep) ){
                                            new_dep.add(dep);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (new_dep.size() >0){
            dependencies.addAll(new_dep);
            for (String i : new_dep){
                if (i != feature){
                    ArrayList<String> tmp_dep = extract_activity_dependencies_PDG(dependencies,i);
                    if(tmp_dep.size() >0) {
                        for (String s : tmp_dep) {
                            if (!dependencies.contains(s)) {
                                dependencies.add(s);
                            }
                        }
                    }
                }
            }
        }
        if(!dependencies.contains(feature)) {
            dependencies.add(feature);
        }
        Set<String> foo = new HashSet<String>(dependencies);
        ArrayList<String> mainList = new ArrayList<String>();
        mainList.addAll(foo);
        return mainList;
    }

    public   ArrayList<String> extract_activity_dependencies_PDG_soft(ArrayList<String> dependencies, String feature){
        ArrayList<String> new_dep = new ArrayList<String>();
        if(!dependencies.contains(feature)){
            dependencies.add(feature);
        }
        for (SootClass c : Scene.v().getApplicationClasses()) {
            if(c.getName().contains(feature) && !c.getName().contains("$")){
                Chain<SootClass> interfaces = c.getInterfaces();
                Chain<SootField> fields = c.getFields();
                List<SootMethod> methods = c.getMethods();

                if(c.hasSuperclass()){
                    SootClass superclass = c.getSuperclass();
                    if (!this.isExcludeClass(superclass)) {
                        new_dep.add(superclass.getName());
                    }
                }


                for(Iterator<SootField> iter = fields.iterator(); iter.hasNext();){
                    SootField field= iter.next();
                    if (!this.isExcludeClass(field.getType().toString()) && !field.getType().toString().contains("[]")) {
                        if (!dependencies.contains(field.getType().toString()) && !new_dep.contains(field.getType().toString())) {
                            new_dep.add(field.getType().toString());
                        }
                    }

                }
                //handle interfaces
                for(Iterator<SootClass> iter = interfaces.iterator(); iter.hasNext();){
                    SootClass field= iter.next();
                    if (!this.isExcludeClass(field.getType().toString())) {
                        if (!dependencies.contains(field.getType().toString()) && !new_dep.contains(field.getType().toString())) {
                            new_dep.add(field.getType().toString());
                        }
                    }

                }

                for (SootMethod m : methods ){
                    if (!m.hasActiveBody()) {
                        continue;
                    }
                    Body b = m.getActiveBody();
                    boolean able =true;
                    try{
                        ExceptionalUnitGraph CFG = new ExceptionalUnitGraph(b);
                        HashMutablePDG PDG = new HashMutablePDG(CFG);
                        for(Iterator <PDGNode> iter = PDG.iterator(); iter.hasNext();){
                            PDGNode node = iter.next();
                            for (PDGNode a : node.getDependents()){
                                Block block = (Block) a.getNode();
                                for (Iterator<Unit> iter_u = block.iterator(); iter_u.hasNext();){
                                    Unit unit = iter_u.next();
                                    for (ValueBox v : unit.getUseAndDefBoxes()){
                                        String tmp_feat =v.getValue().getType().toString();
                                        if(!dependencies.contains(tmp_feat) && !this.isExcludeClass(tmp_feat) && !new_dep.contains(tmp_feat)  && !tmp_feat.contains("[")){
                                            new_dep.add(tmp_feat);
                                        }
                                        String tmp =v.getValue().toString();
                                        if (tmp.startsWith("class")){
                                            String dep = tmp.split("\"")[1].split(";")[0].substring(1).replace("/",".");
                                            if (!this.isExcludeClass(dep) && !dep.contains("[]") && dep!= feature){
                                                if(!dep.equals(c.getName()) ){
                                                    if(!dependencies.contains(dep) && !new_dep.contains(dep) ){
                                                        new_dep.add(dep);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                            }
                        }
                    }catch(Exception e){
                        able = false;
                    }
                    if (able == false){
                        if(Instrumenter.DEBUG) {
                            System.out.println("Impossible to extract Block\nTrying Manually...\n");
                        }
                        for (Iterator<Local> iter_local = b.getLocals().iterator(); iter_local.hasNext();){
                            Local local_tmp = iter_local.next();
                            if (!this.isExcludeClass(local_tmp.getType().toString()) && !local_tmp.getType().toString().contains("[]")){
                                if(!local_tmp.getType().toString().equals(c.getName()) ){
                                    if(!dependencies.contains(local_tmp.getType().toString()) && !new_dep.contains(local_tmp.getType().toString())){
                                        new_dep.add(local_tmp.getType().toString());
                                    }
                                }
                            }
                        }
                        for (Iterator<ValueBox> iter_units = b.getUseAndDefBoxes().iterator() ; iter_units.hasNext();){
                            ValueBox value = iter_units.next();
                            String tmp_feat =value.getValue().getType().toString();
                            if(!dependencies.contains(tmp_feat) && !this.isExcludeClass(tmp_feat) && !new_dep.contains(tmp_feat)  && !tmp_feat.contains("[")){
                                new_dep.add(tmp_feat);
                            }
                            String tmp =value.getValue().toString();
                            if (tmp.startsWith("class")){
                                String dep = tmp.split("\"")[1].split(";")[0].substring(1).replace("/",".");
                                if (!this.isExcludeClass(dep) && !dep.contains("[]") && dep!= feature){
                                    if(!dep.equals(c.getName()) ){
                                        if(!dependencies.contains(dep) && !new_dep.contains(dep) ){
                                            new_dep.add(dep);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (new_dep.size() >0){
            dependencies.addAll(new_dep);
            for (String i : new_dep){
                System.out.println(i);
                if (i != feature){
                    ArrayList<String> tmp_dep = extract_activity_dependencies_PDG(dependencies,i);
                    if(tmp_dep.size() >0) {
                        for (String s : tmp_dep) {
                            if (!dependencies.contains(s)) {
                                dependencies.add(s);
                            }
                        }
                    }
                }
            }
        }
        if(!dependencies.contains(feature)) {
            dependencies.add(feature);
        }
        Set<String> foo = new HashSet<String>(dependencies);
        ArrayList<String> mainList = new ArrayList<String>();
        mainList.addAll(foo);
        return mainList;
    }



}
