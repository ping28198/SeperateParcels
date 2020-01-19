#include "yolo_v3.h"
//#include "logger.h"
//Logger logger("D:/log/", LogLevelAll);
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

YOLO_V3::YOLO_V3()
{
	session = NULL;
	input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 544, 544, 3 }));
	m_score_threshold = 0.3;
	m_max_instance_per_class = 1;
	m_initial_state = 0;
}

YOLO_V3::~YOLO_V3()
{
	session->Close();
}

int YOLO_V3::detect_image( const std::string &image_path,
	std::vector<cv::Rect> &detected_box, std::vector<int> &detected_class, 
	std::vector<float> &detected_scores, float score_threshold)
{
	if (image_path.empty()) return -1;
	cv::Mat src_mat = cv::imread(image_path);
	if (src_mat.empty()) return -1;
	return detect_mat(src_mat, detected_box, detected_class, detected_scores, score_threshold);
}

int YOLO_V3::detect_mat(cv::Mat &src_mat,
	std::vector<cv::Rect> &detected_box, std::vector<int> &detected_class, 
	std::vector<float> &detected_scores, float score_threshold)
{
	if (src_mat.empty()) return -1;
	if (m_initial_state == 0) return -1;
	cv::Size input_image_size(src_mat.cols, src_mat.rows);
	cv::Size input_shape(544, 544);
	cv::Mat resized_mat;
	if (src_mat.channels() == 1)
	{
		cv::cvtColor(src_mat, src_mat, cv::COLOR_GRAY2RGB);//输入图像为灰度，
	}
	else
	{
		cv::cvtColor(src_mat, src_mat, cv::COLOR_BGR2RGB);//输入图像为彩色
	}

	resize_image_pad(src_mat, input_shape, resized_mat);
	//imwrite("d:/test_img3.jpg", resized_mat);
	//tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({ 1, 544, 544, 3 }));
	prepare_image_data(resized_mat,  input_shape, &input_tensor);
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
	{ "input_1", input_tensor }
	};
	// 输出outputs
	std::vector<tensorflow::Tensor> outputs;
	
	//std::vector<std::string> test_output_nodes;
	//test_output_nodes.push_back("conv2d_13/BiasAdd:0");
	// 运行会话，最终结果保存在outputs中
	tensorflow::Status status = session->Run(inputs, { output_nodes }, {}, &outputs);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return -1;
	}
	else {
		//std::cout << "Run session successfully" << endl;
	}

	float applied_score_threshold = (score_threshold < 0.0001) ? this->m_score_threshold : score_threshold;




	//////////////////////////////////////////////////////////////////////////
	//调试
	//auto tf_test_tensor = outputs[0];
	//int dims = tf_test_tensor.shape().dims();
	//auto test_tensor = tf_test_tensor.tensor<float, 4>();
	//auto eig_tensor = Eigen::Tensor<float, 4, 1>(test_tensor);
	//int num_all_dims=1;
	//for (int n=0;n<dims;n++)
	//{
	//	num_all_dims *= eig_tensor.dimension(n);
	//}
	//Eigen::array<Eigen::Index, 2> shap_t{ num_all_dims, 1 };
	//Eigen::Tensor<float, 2, 1> reshape_t = eig_tensor.reshape(shap_t);
	//std::ofstream testfile("d:/test_out.txt", std::ios::out);
	//if (testfile)
	//{
	//	char tmpstr[64] = { 0 };
	//	for (int m = 0; m < num_all_dims; m++)
	//	{
	//		sprintf(tmpstr, "%.18f", reshape_t(m, 0));
	//		testfile << tmpstr << std::endl;
	//	}
	//	//testfile << reshape_t << std::endl;
	//	testfile.flush();
	//	testfile.close();
	//}
	//else
	//{
	//	std::cout << "文件打开失败" << std::endl;
	//}


	//return 0;

	int num_class = 2; //类别数量
	float iou_threshold = 0.45;
	std::vector<cv::Rect> mDetectBoxes;
	std::vector<float> mDetectScores;
	std::vector<int> mDetectClasses;


	yolo_eval(outputs, anchors, num_class, input_shape, m_max_instance_per_class, applied_score_threshold, iou_threshold,
		mDetectBoxes, mDetectScores, mDetectClasses);


	//////////////////////////////////////////////////////////////////////////
	//////测试
	//char score_str[32] = { 0 };
	//for (int i = 0; i < mDetectBoxes.size(); i++)
	//{
	//	cv::Rect s_rec = mDetectBoxes[i];
	//	cv::Rect img_rect(0, 0, input_shape.width, input_shape.height);

	//	if (crop_rect(img_rect, s_rec))
	//	{
	//		cv::Scalar clor;
	//		if (mDetectClasses[i] == 0)
	//		{
	//			clor = cv::Scalar(255, 0, 0);
	//			std::cout << "class:0 scores:" << mDetectScores[i] << " boxes:" << mDetectBoxes[i].y << " " << mDetectBoxes[i].x << " "
	//				<< mDetectBoxes[i].y + mDetectBoxes[i].height << " " << mDetectBoxes[i].x + mDetectBoxes[i].width << " " << std::endl;

	//		}
	//		else
	//		{
	//			clor = cv::Scalar(0, 255, 0);
	//			std::cout << "class:1 scores:" << mDetectScores[i] << " boxes:" << mDetectBoxes[i].y << " " << mDetectBoxes[i].x << " "
	//				<< mDetectBoxes[i].y + mDetectBoxes[i].height << " " << mDetectBoxes[i].x + mDetectBoxes[i].width << " " << std::endl;

	//		}
	//		cv::rectangle(resized_mat, s_rec, clor, 1);
	//		std::sprintf(score_str, "%d:%f", mDetectClasses[i], mDetectScores[i]);
	//		cv::putText(resized_mat, score_str, cv::Point(s_rec.x, s_rec.y), 1, 1, clor);
	//	}
	//	else
	//	{
	//		printf("超出图像范围\n");
	//	}

	//}
	//imshow("box", resized_mat);
	////结束测试

	for (int i = 0; i < mDetectBoxes.size(); i++)
	{
		cv::Rect s_rec = mDetectBoxes[i];
		float scal_w = input_image_size.width / float(input_shape.width);
		float scal_h = input_image_size.height / float(input_shape.height);
		cv::Rect d_rec;
		float scal_ = std::max(scal_w, scal_h);
		//cv::Rect d_rec(s_rec.x * scal_, s_rec.y * scal_, s_rec.width * scal_, s_rec.height * scal_);
		d_rec.x = scal_*s_rec.x; d_rec.y = scal_*s_rec.y; d_rec.width = scal_*s_rec.width; d_rec.height = scal_*s_rec.height;
		if (scal_w > scal_h)
		{
			d_rec.y -= (input_shape.height*scal_ - input_image_size.height) / 2;
		}
		else
		{
			d_rec.x -= (input_shape.width*scal_ - input_image_size.width) / 2;
		}
		cv::Rect img_rect(0, 0, input_image_size.width, input_image_size.height);

		if (crop_rect(img_rect, d_rec))
		{
			detected_box.push_back(d_rec);
			detected_scores.push_back(mDetectScores[i]);
			detected_class.push_back(mDetectClasses[i]);
		}
		else
		{
			printf("超出图像范围\n");
		}

	}






	return detected_box.size();

}

//std::string model_path, float score_threshold, int max_instance_per_class
int YOLO_V3::initial(std::string model_path, float score_threshold, int max_instance_per_class)
{
	if (session != NULL) return 1;
	//logger.TraceInfo("enter_initial");

	tensorflow::Status status = NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 0;
	}
	else {
		std::cout << "Session created successfully" << std::endl;
	}
	//logger.TraceInfo("new_session_over");

	// 读取二进制的模型文件到graph中
	status = ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 0;
	}
	else {
		std::cout << "Load graph protobuf successfully" << std::endl;
	}
	//logger.TraceInfo("read_model_over");

	// 将graph加载到session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 0;
	}
	else {
		std::cout << "Add graph to session successfully" << std::endl;
	}
	//logger.TraceInfo("create_graph_over");

	if (score_threshold > 0.0001)
	{
		this->m_score_threshold = score_threshold;
	}
	if (max_instance_per_class>=1)
	{
		this->m_max_instance_per_class = max_instance_per_class;
	}
	output_nodes.push_back("conv2d_10/BiasAdd");
	output_nodes.push_back("conv2d_13/BiasAdd");
	anchors.push_back(cv::Size(106, 107));//keras_yolo_作者qqwwee的anchor设计，注意并不是原始版本
	anchors.push_back(cv::Size(176, 221));
	anchors.push_back(cv::Size(344, 319));
	//anchors.push_back(cv::Size(13, 18));
	anchors.push_back(cv::Size(30, 35));
	anchors.push_back(cv::Size(48, 76));
	anchors.push_back(cv::Size(106, 107));
	m_initial_state = 1;
	//logger.TraceInfo("return yolo initial");
	return 1;
}

int YOLO_V3::resize_image_pad(cv::Mat srcimg, cv::Size msize, cv::Mat &dstimg)
{
	int input_height = msize.height;
	int input_width = msize.width;
	cv::Mat resied_m = cv::Mat(msize, srcimg.type(), cv::Scalar(128, 128, 128));
	double fscal_y = double(input_height) / srcimg.rows;
	double fscal_x = double(input_width) / srcimg.cols;
	double fscal = std::min(fscal_x, fscal_y);
	cv::Size rsize;
	rsize.height = fscal * srcimg.rows;
	rsize.width = fscal * srcimg.cols;
	cv::Mat resize_src;
	cv::resize(srcimg, resize_src, rsize,0,0,cv::INTER_AREA); //缩小不可用其他方法
	cv::Rect imRect;
	imRect.x = (input_width - resize_src.cols) / 2;
	imRect.y = (input_height - resize_src.rows) / 2;
	imRect.width = resize_src.cols;
	imRect.height = resize_src.rows;
	//imRect.x = (imRect.x < 0) ? 0 : imRect.x;
	//imRect.y = (imRect.y < 0) ? 0 : imRect.y;
	//imRect.width = (imRect.width > input_width - imRect.x) ? input_width - imRect.x : imRect.width;
	//imRect.height = (imRect.height > input_height - imRect.y) ? input_height - imRect.y : imRect.height;
	resize_src.copyTo(resied_m(imRect));
	dstimg = resied_m;
	return 1;
}

int YOLO_V3::yolo_eval(std::vector<tensorflow::Tensor> inputs, std::vector<cv::Size> anchors,
	int num_class, cv::Size image_shape, size_t max_box, float score_threshold, 
	float iou_threshold, std::vector<cv::Rect> &box_detected, std::vector<float> &box_scores,
	std::vector<int> &class_detected)
{
	int num_layers = inputs.size();
	std::vector<Eigen::Tensor<float, 2, 1>> boxes_vec, scores_vec;
	for (size_t i = 0; i < num_layers; i++)
	{
		std::vector<cv::Size> anc_l;
		int ind_s = i * (anchors.size() / num_layers);
		int ind_e = (i + 1) * (anchors.size() / num_layers);
		anc_l.assign(anchors.begin() + ind_s, anchors.begin() + ind_e);

		yolo_eval_layer(inputs[i], anc_l, num_class,
			image_shape, boxes_vec, scores_vec);
	}
	//Eigen::Tensor<float, 2, 1> all_boxes , all_scores;
	Eigen::Tensor<float, 2, 1> all_boxes = boxes_vec[0].concatenate(boxes_vec[1], 0);
	Eigen::Tensor<float, 2, 1> all_scores = scores_vec[0].concatenate(scores_vec[1], 0);



	//////////////////////////////////////////////////////////////////////////
//调试
	//std::string f="d:/x_a.txt";
	//int numele = all_scores.dimension(0);
	////Eigen::array<Eigen::Index, 2> shap_t{ dim_1*grid_x*grid_y*num_anchors, 2 };
	//Eigen::Tensor<float, 2, 1> test_resp = all_scores;
	//std::ofstream testfile(f.c_str(), std::ios::out);
	//if (testfile)
	//{
	//	char tmpstr[64] = { 0 };
	//	for (int m = 0; m < numele; m++)
	//	{
	//		sprintf(tmpstr, "%.18f", test_resp(m, 0));
	//		testfile << tmpstr;
	//		sprintf(tmpstr, " %.18f", test_resp(m, 1));
	//		testfile << tmpstr << std::endl;
	//	}
	//	//testfile << reshape_t << std::endl;
	//	testfile.flush();
	//	testfile.close();
	//}
	//else
	//{
	//	std::cout << "文件打开失败" << std::endl;
	//}
	//if (num_layers==1)
	//{
	//	all_boxes = boxes_vec[0];
	//	all_scores = scores_vec[0];
	//}
	//else if (num_layers == 2)
	//{
	//	all_boxes = boxes_vec[0].concatenate(boxes_vec[1],0);
	//	all_scores = scores_vec[0].concatenate(scores_vec[1], 0);
	//}
	//else if(num_layers ==3)
	//{
	//	Eigen::Tensor<float, 2, 1> box_1 = (boxes_vec[0].concatenate(boxes_vec[1], 0));
	//	all_boxes = box_1.concatenate(boxes_vec[2], 0);
	//	Eigen::Tensor<float, 2, 1> scores_1 = scores_vec[0].concatenate(scores_vec[1], 0);
	//	all_scores = scores_1.concatenate(scores_vec[2], 0);
	//}

	//cout<< scores_vec[1] << endl;

	//Eigen::Tensor<bool, 2, 1> mask_t = all_scores >= score_threshold;
	//Eigen::array<int, 1> _dims{1};
	//Eigen::Tensor<float, 1, 1> max_op = all_scores.maximum(_dims);
	//auto mask_box_t = mask_t.reduce(1, max_op);

	//将box 转换为像素级
	assert(all_boxes.dimension(0) != all_scores.dimension(0));
	//size_t num_abox = all_boxes.dimension(0);
	//Eigen::Tensor<float, 2, 1> box_size_t(1, 4);
	//box_size_t(1, 0) = (float)image_shape.width;
	//box_size_t(1, 1) = (float)image_shape.height;
	//box_size_t(1, 2) = (float)image_shape.width;
	//box_size_t(1, 3) = (float)image_shape.height;
	//Eigen::array<Eigen::Index, 2> broad{ num_abox,1 };
	//Eigen::Tensor<float, 2, 1> all_boxes_int = all_boxes;


	int num_abox = all_boxes.dimension(0);
	//提取置信度大于threshold的box
	std::vector<std::vector<cv::Rect>> candi_box_vec;
	std::vector < std::vector<float>> candi_socres_vec;
	std::vector < std::vector<int>> candi_class_vec;
	for (size_t i = 0; i < num_class; i++)
	{
		//Eigen::array<Eigen::Index, 2> slice_start{0,i};
		//Eigen::array<Eigen::Index, 2> slice_end{num_abox,1};
		//Eigen::Tensor<float, 2, 1> sliced_t = all_scores.slice(slice_start, slice_end);
		std::vector<cv::Rect> b_vec;
		std::vector<float> s_vec;
		std::vector<int> c_vec;
		cv::Rect r;
		for (size_t j = 0; j < num_abox; j++)
		{
			if (all_scores(j, i) >= score_threshold)
			{
				r.x = round(all_boxes(j, 0) - all_boxes(j, 2)/2.0);
				r.y = round(all_boxes(j, 1) - all_boxes(j, 3)/2.0);
				r.width = round(all_boxes(j, 2) );
				r.height = round(all_boxes(j, 3) );
				b_vec.push_back(r);
				s_vec.push_back(all_scores(j, i));
				c_vec.push_back(i);
			}
		}
		candi_box_vec.push_back(b_vec);
		candi_socres_vec.push_back(s_vec);
		candi_class_vec.push_back(c_vec);
	}



	for (int i = 0; i < candi_box_vec.size(); i++)
	{
		cv::dnn::experimental_dnn_34_v7::MatShape index_res;

		cv::dnn::NMSBoxes(candi_box_vec[i], candi_socres_vec[i], score_threshold, iou_threshold,
			index_res, 1.0, max_box);

		for (size_t j = 0; j < index_res.size(); j++)
		{
			int ind = index_res[j];
			box_detected.push_back(candi_box_vec[i][ind]);
			box_scores.push_back(candi_socres_vec[i][ind]);
			class_detected.push_back(candi_class_vec[i][ind]);
		}
		//box_detected.insert(box_detected.end(),candi_box_vec[i].begin(), candi_box_vec[i].end());
		//box_scores.insert(box_scores.end(), candi_socres_vec[i].begin(), candi_socres_vec[i].end());
		//class_detected.insert(class_detected.end(), candi_class_vec[i].begin(), candi_class_vec[i].end());
	}
	//cv::dnn::NMSBoxes(,)
	return 1;


}


int YOLO_V3::yolo_eval_layer(tensorflow::Tensor input_t, std::vector<cv::Size> anchors, 
	int num_class, cv::Size input_shape, std::vector<Eigen::Tensor<float, 2, 1>> &boxes, 
	std::vector<Eigen::Tensor<float, 2, 1>> &scores)
{
	long grid_x = input_t.shape().dim_size(1);
	long grid_y = input_t.shape().dim_size(2);
	long num_anchors = anchors.size();
	long dims = input_t.shape().dims();
	long channels = input_t.shape().dim_size(0);
	long feature_length = input_t.shape().dim_size(3);
	long all_dims = 1;
	for (int i = 0; i < dims; i++)
	{
		all_dims *= input_t.shape().dim_size(i);
	}
	long dim_1 = all_dims / (grid_x*grid_y*(num_class + 5)*num_anchors);
	auto input_tensor_mapped = input_t.tensor<float, 4>(); //tensormap
	auto src_tensor  = Eigen::Tensor<float, 4, Eigen::RowMajor>(input_tensor_mapped); //转为tensor


	std::array<Eigen::Index, 5> reshape_dims{ dim_1, grid_x, grid_y, num_anchors,num_class + 5 };
	Eigen::Tensor<float, 5, Eigen::RowMajor> reshape_t = src_tensor.reshape(reshape_dims);

	//修正confidence
	std::array<Eigen::Index, 5> slice_start{ 0 , 0, 0, 0, 4 };
	std::array<Eigen::Index, 5> slice_end{ dim_1, grid_x, grid_y, num_anchors, 1 };
	auto slice_t = reshape_t.slice(slice_start, slice_end);
	Eigen::Tensor<float, 5, 1> confidence_t = slice_t.sigmoid();

	//修正宽高
	//////////////////////////////////////////////////////////////////////////
	//float scale_ = 32.0 / (input_shape.width / grid_x);
	Eigen::Tensor<float, 5, 1> anc(1, 1, 1, num_anchors, 2);
	for (int i = 0; i < num_anchors; i++)
	{
		anc(0, 0, 0, i, 0) = (float)anchors[i].width;
		anc(0, 0, 0, i, 1) = (float)anchors[i].height;
	}
	//cout << anc << endl;
	std::array<Eigen::Index, 5> bcast{ dim_1, grid_x, grid_y ,1,1 };
	Eigen::Tensor<float, 5, 1> wh_anchor_t = anc.broadcast(bcast);
	//Eigen::Tensor<float, 5, 1> anc_1(1,1,1,1,2);
	//anc1(0,0,0,0,0) 
	slice_start[4] = 2;
	slice_end[4] = 2;
	Eigen::Tensor<float, 5, 1> slice_wh_t = reshape_t.slice(slice_start, slice_end);
	Eigen::Tensor<float, 5, 1> wh_t = slice_wh_t.exp()*wh_anchor_t;









	//修正位置
	std::array<Eigen::Index, 5> bcast2{ 1,1,1,num_anchors,1 };
	Eigen::Tensor<float, 5, 1> an_x(1, grid_x, 1, 1, 1);
	for (int i = 0; i < grid_x; i++)
	{
		an_x(0, i, 0, 0, 0) = (float)i;
	}
	bcast[0] = 1; bcast[1] = 1; bcast[2] = grid_y; bcast[3] = 1; bcast[4] = 1;
	//bcast2[0] = 1; bcast2[1] = 1; bcast2[2] = 1; bcast2[3] = num_anchors; bcast2[4] = 1;
	Eigen::Tensor<float, 5, 1> xy_x_t = an_x.broadcast(bcast);
	Eigen::Tensor<float, 5, 1> xy_x_t_1 = xy_x_t.broadcast(bcast2);

	Eigen::Tensor<float, 5, 1> an_y(1, 1, grid_y, 1, 1);
	for (int i = 0; i < grid_y; i++)
	{
		an_y(0, 0, i, 0, 0) = (float)i;
	}
	bcast[1] = grid_x;
	bcast[2] = 1;
	Eigen::Tensor<float, 5, 1> xy_y_t = an_y.broadcast(bcast);
	Eigen::Tensor<float, 5, 1> xy_y_t_1 = xy_y_t.broadcast(bcast2);
	Eigen::Tensor<float, 5, 1> con_xy_grid = xy_y_t_1.concatenate(xy_x_t_1, 4);

	slice_start[4] = 0;
	slice_end[4] = 2;
	auto slice_xy_t = reshape_t.slice(slice_start, slice_end);
	Eigen::Tensor<float, 5, 1> xy_sigmoid_t = slice_xy_t.sigmoid();
	Eigen::Tensor<float, 5, 1> xy_add = xy_sigmoid_t + con_xy_grid;

	Eigen::Tensor<float, 5, 1> grid_xy(1, 1, 1, 1, 2);
	grid_xy(0, 0, 0, 0, 0) = grid_x;
	grid_xy(0, 0, 0, 0, 1) = grid_y;
	bcast[0] = dim_1; bcast[1] = grid_x; bcast[2] = grid_y; bcast[3] = num_anchors; bcast[4] = 1;
	auto grid_xy_bc = grid_xy.broadcast(bcast);
	Eigen::Tensor<float, 5, 1> xy_div = xy_add / grid_xy_bc;

	grid_xy(0, 0, 0, 0, 0) = (float)input_shape.width;
	grid_xy(0, 0, 0, 0, 1) = (float)input_shape.height;
	Eigen::Tensor<float, 5, 1> img_wh_bc = grid_xy.broadcast(bcast);
	Eigen::Tensor<float, 5, 1> xy_t = xy_div * img_wh_bc;


	//获得class类别
	slice_start[4] = 5;
	slice_end[4] = num_class;
	Eigen::Tensor<float, 5, 1> class_t = reshape_t.slice(slice_start, slice_end).sigmoid();











	//展开矩阵
	//box
	Eigen::Tensor<float, 5, 1> box_xy_wh = xy_t.concatenate(wh_t, 4);
	std::array<Eigen::Index, 2> shape_2{ dim_1*grid_x*grid_y*num_anchors,4 };
	Eigen::Tensor<float, 2, 1> box_xy_wh_1 = box_xy_wh.reshape(shape_2);

	bcast[0] = 1; bcast[1] = 1; bcast[2] = 1; bcast[3] = 1; bcast[4] = num_class;
	shape_2[1] = num_class;
	Eigen::Tensor<float, 5, 1> confidence_b_t = confidence_t.broadcast(bcast);
	Eigen::Tensor<float, 5, 1> class_scores_t = confidence_b_t * class_t;
	Eigen::Tensor<float, 2, 1> class_scores_1 = class_scores_t.reshape(shape_2);

//	//////////////////////////////////////////////////////////////////////////
////调试
//	std::string f;
//	if (grid_y == 17)
//	{
//		f = "d:/x1.txt";
//	}
//	else
//	{
//		f = "d:/x2.txt";
//	}
//
//	Eigen::array<Eigen::Index, 2> shap_t{ dim_1*grid_x*grid_y*num_anchors, 2 };
//	Eigen::Tensor<float, 2, 1> test_resp = class_scores_1;
//	std::ofstream testfile(f.c_str(), std::ios::out);
//	if (testfile)
//	{
//		char tmpstr[64] = { 0 };
//		for (int m = 0; m < dim_1*grid_x*grid_y*num_anchors; m++)
//		{
//			sprintf(tmpstr, "%.18f", test_resp(m, 0));
//			testfile << tmpstr;
//			sprintf(tmpstr, " %.18f", test_resp(m, 1));
//			testfile << tmpstr << std::endl;
//		}
//		//testfile << reshape_t << std::endl;
//		testfile.flush();
//		testfile.close();
//	}
//	else
//	{
//		std::cout << "文件打开失败" << std::endl;
//	}






	////////////////////////////////////////////////////////////////////////
	//调试用
	//std::cout << xy_t.dimension(0) << ":";
	//std::cout << xy_t.dimension(1) << ":";
	//std::cout << xy_t.dimension(2) << ":";
	//std::cout << xy_t.dimension(3) << ":";
	//std::cout << xy_t.dimension(4) << ":" << std::endl;
	//std::array<Eigen::Index, 2> test_arr{ dim_1*grid_x*grid_y*num_anchors*(num_class+5),1 };
	//Eigen::Tensor<float, 2, 1> test_t = src_tensor.reshape(test_arr);
	//std::string f;
	//if (grid_y == 17)
	//{
	//	f = "d:/x1.txt";
	//}
	//else
	//{
	//	f = "d:/x2.txt";
	//}
	//std::ofstream testfile(f.c_str(), std::ios::out);
	//if (testfile)
	//{
	//	testfile << test_t << std::endl;
	//	testfile.flush();
	//	testfile.close();
	//}
	//else
	//{
	//	std::cout << "文件打开失败" << std::endl;
	//}
	////

	boxes.push_back(box_xy_wh_1);
	scores.push_back(class_scores_1);
	return 1;

}

int YOLO_V3::yolo_eval_layer_tf(tensorflow::Tensor input_t, std::vector<cv::Size> anchors, int num_class, cv::Size input_shape, std::vector<Eigen::Tensor<float, 2, 1>> &boxes, std::vector<Eigen::Tensor<float, 2, 1>> &scores)
{
	using namespace tensorflow;
	using namespace tensorflow::ops;
	int num_anchors = anchors.size();
	auto in_shape_tf = input_t.shape();
	Scope root = Scope::NewRootScope();
	








	return 1;
}

int YOLO_V3::crop_rect(cv::Rect main_rect, cv::Rect &to_crop_rect)
{
	if (to_crop_rect.x + to_crop_rect.width <= main_rect.x) return 0;
	if (to_crop_rect.y + to_crop_rect.height <= main_rect.y) return 0;
	if (to_crop_rect.x >= main_rect.width + main_rect.x) return 0;
	if (to_crop_rect.y >= main_rect.height + main_rect.y) return 0;
	if (main_rect.x > to_crop_rect.x) to_crop_rect.x = main_rect.x;
	if (main_rect.y > to_crop_rect.y) to_crop_rect.y = main_rect.y;
	if (main_rect.x + main_rect.width < to_crop_rect.x + to_crop_rect.width)
	{
		to_crop_rect.width = main_rect.x + main_rect.width - to_crop_rect.x;
	}
	if (main_rect.y + main_rect.height < to_crop_rect.y + to_crop_rect.height)
	{
		to_crop_rect.height = main_rect.y + main_rect.height - to_crop_rect.y;
	}
	return 1;
}

int YOLO_V3::prepare_image_data(cv::Mat src_img, cv::Size input_shape, tensorflow::Tensor * dst_tensor)
{
	if (src_img.empty()) return 0;

	int channels = 3;
	
	//cv::Mat resized_img;//输入模型的图像数据
	//resize_image_pad(src_img, input_shape, resized_img);
	//imshow("resized", resized_img);
	//int channels_ = resized_img.channels();
	//dstMat = resized_img;
	//准备数据
	//tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({ 1, input_height, input_width, channels }));
	const tensorflow::uint8* source_data = src_img.data;
	auto input_tensor_mapped = dst_tensor->tensor<float, 4>();

	for (int i = 0; i < input_shape.width; i++) {
		const tensorflow::uint8* source_row = source_data + (i * input_shape.width * channels);
		for (int j = 0; j < input_shape.height; j++) {
			const tensorflow::uint8* source_pixel = source_row + (j * channels);
			for (int c = 0; c < channels; c++) {
				const tensorflow::uint8* source_value = source_pixel + c;
				input_tensor_mapped(0, i, j, c) = ((float)(*source_value)) / 255.0;
			}
		}
	}
	return 1;
}

