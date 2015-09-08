set -e

#Get nlopt
wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz
tar -xvzf nlopt-2.4.2.tar.gz

NLOPT_PREFIX=$HOME/feedback-kde/nlopt-2.4.2

#Compile and install nlopt
cd nlopt-2.4.2
patch -p1 < ../nlopt-patch.diff
./configure --prefix=$NLOPT_PREFIX
make clean
make -j 8
make install
cd ..


export CPPFLAGS="-I$NLOPT_PREFIX/include"
export LDFLAGS="-L$NLOPT_PREFIX/lib/"
#Compile and install postgres
mkdir -p pgsql
./configure -with-opencl --prefix=$HOME/feedback-kde/pgsql
make clean
make -j 8
make install

PGBIN=$HOME/feedback-kde/pgsql/bin

#Setup
$PGBIN/initdb ./experiments
$PGBIN/pg_ctl -D ./experiments -l logfile start
sleep 3
$PGBIN/dropdb --if-exists experiments
$PGBIN/createdb experiments
cp ./analysis/conf.sh.template ./analysis/conf.sh
bash ./analysis/static/datasets/prepare.sh
bash ./analysis/static/timing/prepare_experiment.sh
$PGBIN/pg_ctl -D ./experiments -l logfile stop

#Insert sanity check here.
