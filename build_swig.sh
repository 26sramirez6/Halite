rm -rf swig_wrap.o halite_swig.so swig_wrap.cxx
swig -c++ -python swig.i
/usr/local/bin/c++ -fPIC -I/home/saul.ramirez/anaconda3/include/python3.7m -I/home/saul.ramirez/eclipse-workspace/Halite -c swig_wrap.cxx
/usr/local/bin/c++ -fPIC -shared swig_wrap.o -o halite_swig.so
