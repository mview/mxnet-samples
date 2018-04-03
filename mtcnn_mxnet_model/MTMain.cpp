#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sys/time.h>
// opencv
#include <opencv2/opencv.hpp>
#include <engine.h>
#include <mxnet-cpp/MxNetCpp.h>
using namespace std;
using namespace mxnet::cpp;

typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score; /**< Larger score should mean higher confidence. */
} FaceRect;

typedef struct FacePts {
  float x[5],y[5];
} FacePts;

typedef struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts facePts;
  double roll;
  double pitch;
  double yaw;
} FaceInfo;

DeviceType devicetype_=mxnet::cpp::kCPU;
void set_mode(DeviceType type)
{
    devicetype_=type;
}
class Blob:public NDArray {
public:
    Blob(const Shape &shape, const Context &context=Context::cpu()):
       NDArray(shape,context){
    }
    Blob(const std::vector<mx_uint> shape, const mxnet::cpp::Context &context=Context::cpu()):
       mxnet::cpp::NDArray(shape,context){
    }
    inline mx_float *cpu_data() const {
        return (mx_float *)GetData();
    }
    inline size_t count() const {
        return NDArray::Size();
    }
    inline size_t nums() const{
        return GetShape()[0];
    }
    inline size_t channels() const{
        return GetShape()[1];
    }
    inline size_t width() const{
        return GetShape()[2];
    }
    inline size_t height() const{
        return GetShape()[3];
    }
    inline std::vector<mx_uint> GetShape() const{
        return NDArray::GetShape();
    }
    inline void Reshape(int num,int ch,int width,int height){
        NDArray::Reshape(Shape(num, ch, width, height)); 
    }
    inline void AddNDArrayVector(std::vector<NDArray> data_vector){
        int w=width();
        int h=height();
        int ch=channels();
        int num=nums();
        int size=count()/num;
        Blob tmp(Shape(num,ch,w,h),Context::cpu());
        for (int i=0;i<data_vector.size();++i) {
            memcpy(&tmp.cpu_data()[i*size],data_vector[i].GetData(),size*sizeof(float));

        }
        tmp.CopyTo(this);
    }
    virtual ~Blob(){
    }
};
class Net{
public:
    Net(const std::string &symbol_name,const std::string &para_name,Shape shape=Shape(0,0,0,0))
    {
        Symbol net;
        net_   = Symbol::Load(symbol_name);
        LoadParameters(GetContext(),para_name);
        executor_ =NULL;
        if (shape[2] && shape[3]) {
            args_map_["data"] = NDArray(shape,GetContext());
            executor_ = net_.SimpleBind(GetContext(), args_map_);
        }
    }
    Context  GetContext() const{
        if (devicetype_==mxnet::cpp::kGPU) {
            return Context::gpu();
        }
        return Context::cpu(); 
    }
    const std::vector<mx_uint> & shape() {
        std::vector<mx_uint>  test=args_map_["data"].GetShape();
       return args_map_["data"].GetShape();
    }
    void Reshape(){
        //executor_->Reshape();
    }
    void Forward(Blob *input=nullptr){
        bool is_train=false;
        if (input) {
            if(executor_)delete executor_;
            if (input->GetContext().GetDeviceType()==devicetype_) {
                args_map_["data"] = *input; 
            }else{
                Blob data(input->GetShape(),GetContext());
                input->CopyTo(&data);
                args_map_["data"] = data; 
            }
            executor_ = net_.SimpleBind(GetContext(), args_map_);
        }
        executor_->Forward(is_train); 
    }
    inline const std::vector<Blob*> output_blobs(){
        out_blobs.erase( out_blobs.begin(),out_blobs.end() );
        for (auto &output:executor_->outputs) {
            Blob *data=new Blob(output.GetShape(),Context::cpu());
            output.SyncCopyToCPU(data->cpu_data(),data->Size());
            out_blobs.push_back(data); 
        }
        return out_blobs;
    }
    inline const std::vector<Blob*> input_blobs(){
        in_blobs.erase( in_blobs.begin(),in_blobs.end() );
        in_blobs.push_back((Blob*)&args_map_["data"]);
        return in_blobs;
    }

private:
    /*Fill the trained paramters into the model, a.k.a. net, executor*/
    void LoadParameters(Context ctx,string file) {
      map<string, NDArray> paramters;
      NDArray::Load(file, 0, &paramters);
      for (const auto &k : paramters) {
        if (k.first.substr(0, 4) == "aux:") {
          auto name = k.first.substr(4, k.first.size() - 4);
          aux_map_[name] = k.second.Copy(ctx);
        }
        if (k.first.substr(0, 4) == "arg:") {
          auto name = k.first.substr(4, k.first.size() - 4);
          args_map_[name] = k.second.Copy(ctx);
        }
      }
      /*WaitAll is need when we copy data between GPU and the main memory*/
      NDArray::WaitAll();
    }
    map<string, NDArray> args_map_;
    map<string, NDArray> aux_map_;
    Symbol net_;
    Executor * executor_;
    std::vector<Blob*>out_blobs;
    std::vector<Blob*>in_blobs;
};

class MTCNN {
 public:
  MTCNN(const string& proto_model_dir);
  void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo, int minSize, double* threshold, double factor);

 private:
  bool CvMatToNDArraySignalChannel(const cv::Mat& cv_mat, Blob* data);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels,const Blob* input_layer,
          const int height,const int width);
  void SetMean();
  void GenerateBoundingBox( Blob* confidence,Blob* reg,
          float scale,float thresh,int image_width,int image_height);
  void ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        Net* net,double thresh,char netName);
  void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
        Net* net, double thresh, char netName);
  std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes,float thresh,char methodType);
  void Bbox2Square(std::vector<FaceInfo>& bboxes);
  void Padding(int img_w, int img_h);
  std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
  void RegressPoint(const std::vector<FaceInfo>& faceInfo);

 private:
  Net * PNet_;
  Net * RNet_;
  Net * ONet_;

  // x1,y1,x2,t2 and score
  std::vector<FaceInfo> condidate_rects_;
  std::vector<FaceInfo> total_boxes_;
  std::vector<FaceInfo> regressed_rects_;
  std::vector<FaceInfo> regressed_pading_;

  std::vector<cv::Mat> crop_img_;
  int curr_feature_map_w_;
  int curr_feature_map_h_;
  int num_channels_;
};

// compare score
bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
  return a.bbox.score > b.bbox.score;
}

// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Maximum)
std::vector<FaceInfo> MTCNN::NonMaximumSuppression(std::vector<FaceInfo>& bboxes,
                float thresh,char methodType){
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  std::vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1)
      select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;

    FaceRect select_bbox = bboxes[select_idx].bbox;
    float area1 = static_cast<float>((select_bbox.x2-select_bbox.x1+1) * (select_bbox.y2-select_bbox.y1+1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1)
        continue;

      FaceRect& bbox_i = bboxes[i].bbox;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0)
        continue;

      float area2 = static_cast<float>((bbox_i.x2-bbox_i.x1+1) * (bbox_i.y2-bbox_i.y1+1));
      float area_intersect = w * h;

      switch (methodType) {
        case 'u':
          if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
            mask_merged[i] = 1;
          break;
        case 'm':
          if (static_cast<float>(area_intersect) / std::min(area1 , area2) > thresh)
            mask_merged[i] = 1;
          break;
        default:
          break;
        }
    }
  }
  return bboxes_nms;
}

void MTCNN::Bbox2Square(std::vector<FaceInfo>& bboxes){
  for(int i=0;i<bboxes.size();i++){
    float w = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
    float h = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
    float side = h>w ? h:w;
    bboxes[i].bbox.x1 += (w-side)*0.5;
    bboxes[i].bbox.y1 += (h-side)*0.5;

    bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
    bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
    bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
    bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

  }
}

std::vector<FaceInfo> MTCNN::BoxRegress(std::vector<FaceInfo>& faceInfo,int stage){
  std::vector<FaceInfo> bboxes;
  for(int bboxId =0;bboxId<faceInfo.size();bboxId++){
      FaceRect faceRect;
      FaceInfo tempFaceInfo;
      float regh = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
      regh += (stage == 1)? 0:1;
      float regw = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
      regw += (stage == 1)? 0:1;
      faceRect.x1 = faceInfo[bboxId].bbox.x1 + regw * faceInfo[bboxId].regression[0];
      faceRect.y1 = faceInfo[bboxId].bbox.y1 + regh * faceInfo[bboxId].regression[1];
      faceRect.x2 = faceInfo[bboxId].bbox.x2 + regw * faceInfo[bboxId].regression[2];
      faceRect.y2 = faceInfo[bboxId].bbox.y2 + regh * faceInfo[bboxId].regression[3];
      faceRect.score = faceInfo[bboxId].bbox.score;

      tempFaceInfo.bbox = faceRect;
      tempFaceInfo.regression = faceInfo[bboxId].regression;
      if(stage == 3)
        tempFaceInfo.facePts = faceInfo[bboxId].facePts;
      bboxes.push_back(tempFaceInfo);
  }
  return bboxes;
}

// compute the padding coordinates (pad the bounding boxes to square)
void MTCNN::Padding(int img_w,int img_h){
  for(int i=0;i<regressed_rects_.size();i++){
    FaceInfo tempFaceInfo;
    tempFaceInfo = regressed_rects_[i];
    tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_h) ? img_h : regressed_rects_[i].bbox.y2;
    tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_w) ? img_w : regressed_rects_[i].bbox.x2;
    tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 <1) ? 1 : regressed_rects_[i].bbox.y1;
    tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 <1) ? 1 : regressed_rects_[i].bbox.x1;
    regressed_pading_.push_back(tempFaceInfo);
  }
}

void MTCNN::GenerateBoundingBox(Blob* confidence,Blob* reg,
      float scale,float thresh,int image_width,int image_height){
  int stride = 2;
  int cellSize = 12;

  int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0/stride)+1;
  int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0/stride)+1;

  //std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
  int regOffset = curr_feature_map_w_*curr_feature_map_h_;
  // the first count numbers are confidence of face
  int count = confidence->count()/2;
  const float* confidence_data = confidence->cpu_data();
  confidence_data += count;
  const float* reg_data = reg->cpu_data();

  condidate_rects_.clear();
  for(int i=0;i<count;i++){
    if(*(confidence_data+i)>=thresh){
      int y = i / curr_feature_map_w_;
      int x = i - curr_feature_map_w_ * y;

      float xTop = (int)((x*stride+1)/scale);
      float yTop = (int)((y*stride+1)/scale);
      float xBot = (int)((x*stride+cellSize-1+1)/scale);
      float yBot = (int)((y*stride+cellSize-1+1)/scale);
      FaceRect faceRect;
      faceRect.x1 = xTop;
      faceRect.y1 = yTop;
      faceRect.x2 = xBot;
      faceRect.y2 = yBot;
      faceRect.score  = *(confidence_data+i);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[i+0*regOffset],reg_data[i+1*regOffset],reg_data[i+2*regOffset],reg_data[i+3*regOffset]);
      condidate_rects_.push_back(faceInfo);
    }
  }
}

MTCNN::MTCNN(const std::string &proto_model_dir){
  #ifdef CPU_ONLY
    set_mode(mxnet::cpp::kCPU);
  #else
    set_mode(mxnet::cpp::kGPU);
  #endif
  /* Load the network. */
  PNet_ =new Net(proto_model_dir+"det1-symbol.json",proto_model_dir+"det1-0001.params",Shape(0, 0, 0, 0));
  RNet_ =new Net(proto_model_dir+"det2-symbol.json",proto_model_dir+"det2-0001.params",Shape(1, 3, 24, 24));
  ONet_ =new Net(proto_model_dir+"det3-symbol.json",proto_model_dir+"det3-0001.params",Shape(1, 3, 48, 48));
  NDArray input_layer(Shape(1, 3, 48, 48), PNet_->GetContext(), false);
}
void MTCNN::WrapInputLayer(std::vector<cv::Mat>* input_channels,
        const Blob* input_layer, const int height, const int width) {
  float* input_data = input_layer->cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void MTCNN::ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        Net* net,double thresh,char netName){
  int numBox = regressed_rects.size();
  Blob* crop_input_layer = net->input_blobs()[0];
  int input_channels = crop_input_layer->channels();
  int input_width  = crop_input_layer->width();
  int input_height = crop_input_layer->height();
  crop_input_layer->Reshape(1, input_channels, input_width, input_height);
  net->Reshape();

  condidate_rects_.clear();
  // load crop_img data to NDArray
  for(int i=0;i<numBox;i++){
    std::vector<cv::Mat> channels;
    Blob input_layer(net->input_blobs()[0]->GetShape(),Context::cpu());
    WrapInputLayer(&channels,&input_layer,input_width,input_height);
    int pad_top   = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
    int pad_left  = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
    int pad_right = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);
    int pad_bottom= std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);

    cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1-1,regressed_pading_[i].bbox.y2),
                         cv::Range(regressed_pading_[i].bbox.x1-1,regressed_pading_[i].bbox.x2));
    cv::copyMakeBorder(crop_img,crop_img,pad_top,pad_bottom,pad_left,pad_right,cv::BORDER_CONSTANT,cv::Scalar(0));
#ifdef INTER_FAST
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
#else
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_AREA);
#endif
    crop_img = (crop_img-127.5)*0.0078125;
    cv::split(crop_img,channels);

    net->Forward(&input_layer);

    int reg_id = 0;
    int confidence_id = 1;
    if(netName == 'o'){
        reg_id =1;
        confidence_id = 2;
    }
    const Blob* reg = net->output_blobs()[reg_id];
    const Blob* confidence = net->output_blobs()[confidence_id];
    // ONet points_offset != NULL
    const Blob* points_offset = net->output_blobs()[0];

    const float* confidence_data = confidence->cpu_data() + confidence->count()/2;
    const float* reg_data = reg->cpu_data();
    const float* points_data;
    if(netName == 'o') points_data = points_offset->cpu_data();

    if(*(confidence_data) > thresh){
      FaceRect faceRect;
      faceRect.x1 = regressed_rects[i].bbox.x1;
      faceRect.y1 = regressed_rects[i].bbox.y1;
      faceRect.x2 = regressed_rects[i].bbox.x2;
      faceRect.y2 = regressed_rects[i].bbox.y2 ;
      faceRect.score  = *(confidence_data);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[0],reg_data[1],reg_data[2],reg_data[3]);

      // x x x x x y y y y y
      if(netName == 'o'){
        FacePts face_pts;
        float h = faceRect.y2 - faceRect.y1 + 1;
        float w = faceRect.x2 - faceRect.x1 + 1;
        for(int j=0;j<5;j++){
          face_pts.y[j] = faceRect.y1 + *(points_data+j+5) * h - 1;
          face_pts.x[j] = faceRect.x1 + *(points_data+j) * w -1;
        }
        faceInfo.facePts = face_pts;
      }
      condidate_rects_.push_back(faceInfo);
    }
  }
  regressed_pading_.clear();
}

// multi test image pass a forward
void MTCNN::ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        Net* net,double thresh,char netName){
  condidate_rects_.clear();

  int numBox = regressed_rects.size();
  std::vector<NDArray> data_vector;

  Blob* input_data;
  input_data = net->input_blobs()[0];
  int input_width  = input_data->width();
  int input_height = input_data->height();
  int input_channels= input_data->channels();
  Blob mem_data(Shape(numBox,input_channels,input_width,input_height),net->GetContext());


  // load crop_img data to NDArray
  for(int i=0;i<numBox;i++){
    int pad_top   = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
    int pad_left  = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
    int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
    int pad_bottom= std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

    cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1-1,regressed_pading_[i].bbox.y2),
                         cv::Range(regressed_pading_[i].bbox.x1-1,regressed_pading_[i].bbox.x2));
    cv::copyMakeBorder(crop_img,crop_img,pad_left,pad_right,pad_top,pad_bottom,cv::BORDER_CONSTANT,cv::Scalar(0));

#ifdef INTER_FAST
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
#else
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_AREA);
#endif
    crop_img = (crop_img-127.5)*0.0078125;
    Blob data(Shape(1,input_channels,input_width,input_height),Context::cpu());
    CvMatToNDArraySignalChannel(crop_img,&data);
    data_vector.push_back(data);
  }
  regressed_pading_.clear();

  /* extract the features and store */
  mem_data.AddNDArrayVector(data_vector);
  /* fire the network */
  net->Forward(&mem_data);

  int reg_id = 0;
  int confidence_id = 1;
  if(netName == 'o'){
      reg_id =1;
      confidence_id = 2;
  }
  const Blob* reg = net->output_blobs()[reg_id];
  const Blob* confidence = net->output_blobs()[confidence_id];
  // ONet points_offset != NULL
  const Blob* points_offset = net->output_blobs()[0];

  const float* confidence_data = confidence->cpu_data();
  const float* reg_data = reg->cpu_data();
  const float* points_data;
  if(netName == 'o') points_data = points_offset->cpu_data();


  for(int i=0;i<numBox;i++){
    if(*(confidence_data+i*2+1) > thresh){
      FaceRect faceRect;
      faceRect.x1 = regressed_rects[i].bbox.x1;
      faceRect.y1 = regressed_rects[i].bbox.y1;
      faceRect.x2 = regressed_rects[i].bbox.x2;
      faceRect.y2 = regressed_rects[i].bbox.y2 ;
      faceRect.score  = *(confidence_data+i*2+1);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[4*i+0],reg_data[4*i+1],reg_data[4*i+2],reg_data[4*i+3]);

      // x x x x x y y y y y
      if(netName == 'o'){
        FacePts face_pts;
        float h = faceRect.y2 - faceRect.y1 + 1;
        float w = faceRect.x2 - faceRect.x1 + 1;
        for(int j=0;j<5;j++){
          face_pts.y[j] = faceRect.y1 + *(points_data+j+5+10*i) * h - 1;
          face_pts.x[j] = faceRect.x1 + *(points_data+j+10*i) * w -1;
        }
        faceInfo.facePts = face_pts;
      }
      condidate_rects_.push_back(faceInfo);
    }
  }
}
bool MTCNN::CvMatToNDArraySignalChannel(const cv::Mat& cv_mat, Blob* data){
  if (cv_mat.empty())
    return false;

    int size = cv_mat.rows * cv_mat.cols * cv_mat.channels();
    mx_float* image_data= data->cpu_data();
    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    for (int i = 0; i < cv_mat.rows; i++) {
        const float * data = cv_mat.ptr<float>(i);
        for (int j = 0; j < cv_mat.cols; j++) {
            *ptr_image_b++ = *data++;
            *ptr_image_g++ = *data++;
            *ptr_image_r++ = *data++;
        }
    }
    return true;
}

void MTCNN::Detect(const cv::Mat& image,std::vector<FaceInfo>& faceInfo,int minSize,double* threshold,double factor){

  // 2~3ms
  // invert to RGB color space and float type
  cv::Mat sample_single,resized;
  image.convertTo(sample_single,CV_32FC3);

  int height = image.rows;
  int width  = image.cols;
  int minWH = std::min(height,width);
  int factor_count = 0;
  double m = 12./minSize;
  minWH *= m;
  std::vector<double> scales;
  while (minWH >= 12)
  {
    scales.push_back(m * std::pow(factor,factor_count));
    minWH *= factor;
    ++factor_count;
  }

  // 11ms main consum
  //Blob* input_layer = PNet_->input_blobs()[0];
  for(int i=0;i<factor_count;i++)
  {
    double scale = scales[i];
    int hs = std::ceil(height*scale);
    int ws = std::ceil(width*scale);

    // wrap image and normalization using INTER_AREA method
    cv::resize(sample_single,resized,cv::Size(ws,hs),0,0,cv::INTER_AREA);
#ifdef INTER_FAST
    cv::resize(sample_single,resized,cv::Size(ws,hs),0,0,cv::INTER_NEAREST);
#else
    cv::resize(sample_single,resized,cv::Size(ws,hs),0,0,cv::INTER_AREA);
#endif
    resized.convertTo(resized, CV_32FC3, 0.0078125,-127.5*0.0078125);

    // input data
    Blob input_layer(Shape(1, 3, hs, ws),Context::cpu());
    PNet_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels,&input_layer,hs,ws);
    cv::split(resized,input_channels);

    // check data transform right
    PNet_->Forward(&input_layer);

    Blob* reg = PNet_->output_blobs()[0];
    Blob* confidence = PNet_->output_blobs()[1];
    GenerateBoundingBox(confidence, reg, scale, threshold[0],ws,hs);
    std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_,0.5,'u');
    total_boxes_.insert(total_boxes_.end(),bboxes_nms.begin(),bboxes_nms.end());
  }

  int numBox = total_boxes_.size();
  if(numBox != 0){
    total_boxes_ = NonMaximumSuppression(total_boxes_,0.7,'u');
    regressed_rects_ = BoxRegress(total_boxes_,1);
    total_boxes_.clear();

    Bbox2Square(regressed_rects_);
    Padding(width,height);

    /// Second stage
    #ifdef CPU_ONLY
    ClassifyFace(regressed_rects_,sample_single,RNet_,threshold[1],'r');
    #else
    ClassifyFace_MulImage(regressed_rects_,sample_single,RNet_,threshold[1],'r');
    #endif
    condidate_rects_ = NonMaximumSuppression(condidate_rects_,0.7,'u');
    regressed_rects_ = BoxRegress(condidate_rects_,2);

    Bbox2Square(regressed_rects_);
    Padding(width,height);

    /// three stage
    numBox = regressed_rects_.size();
    if(numBox != 0){
      #ifdef CPU_ONLY
      ClassifyFace(regressed_rects_,sample_single,ONet_,threshold[2],'o');
      #else
      ClassifyFace_MulImage(regressed_rects_,sample_single,ONet_,threshold[2],'o');
      #endif
      regressed_rects_ = BoxRegress(condidate_rects_,3);
      faceInfo = NonMaximumSuppression(regressed_rects_,0.7,'m');
    }
  }
  regressed_pading_.clear();
  regressed_rects_.clear();
  condidate_rects_.clear();
}

int main(int argc,char **argv)
{
  if(argc != 3){
    std::cout << "MTMain.bin [model dir] [imagePath]"<<std::endl;
    return 0;
  }
  double threshold[3] = {0.6,0.7,0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);

  std::string imageName = argv[2];
  cv::Mat image = cv::imread(imageName);
  std::vector<FaceInfo> faceInfo;
  clock_t t1 = clock();
  std::cout <<"Detect "<<image.rows<<"X"<<image.cols;
  detector.Detect(image,faceInfo,minSize,threshold,factor);
  #ifdef CPU_ONLY
  std::cout<<" Time Using CPU: " << (clock() - t1)*1.0/1000<<std::endl;
  #else
  std::cout<<" Time Using GPU-CUDNN: " << (clock() - t1)*1.0/1000<<std::endl;
  #endif
  for(int i = 0;i<faceInfo.size();i++){
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
    float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
    cv::rectangle(image,cv::Rect(x,y,w,h),cv::Scalar(255,0,0),2);
  }
  for(int i=0;i<faceInfo.size();i++){
    FacePts facePts = faceInfo[i].facePts;
    for(int j=0;j<5;j++)
      cv::circle(image,cv::Point(facePts.x[j],facePts.y[j]),1,cv::Scalar(255,255,0),2);
  }
  cv::imshow("a",image);
  cv::waitKey(0);

  return 1;
}

