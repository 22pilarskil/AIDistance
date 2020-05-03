
current_dir=$(pwd)

if [ ! -f "$current_dir/frozen_inference_graph.pb" ] ; then 
	echo "\033[0;96m~/Downloading frozen_inference_graph.pb\033[0m"
	wget -O "$current_dir/frozen_inference_graph.pb" \
		"https://www.dropbox.com/s/x617bfgof29rqya/frozen_inference_graph.pb?dl=1" \
		|| echo "\033[0;31~/Error downloading frozen_inference_graph.pb\033[0m"
fi

if [ ! -f "$current_dir/20180402-114759.pb" ] ; then
	echo "\033[0;96m~/Downloading 20180402-114759.pb\033[0m"
	wget -O "$current_dir/20180402-114759.pb" \
		"https://www.dropbox.com/s/p1ab47jcbr5uunr/20180402-114759.pb?dl=1" \
		|| echo "\033[0;31~/Error downloading 20180402-114759.pb\033[0m"
fi
