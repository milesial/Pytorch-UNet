#include <iostream>
#include "Msnhnet/net/MsnhNetBuilder.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhCVUtil.h"

int main(int argc, char** argv) 
{
    std::string msnhnetPath = "unet.msnhnet"; //your msnhnet path
    std::string msnhbinPath = "unet.msnhbin"; //your msnhbin path
    std::string imgPath = "unet.jpg";         //yout test image path
    try
    {
        Msnhnet::NetBuilder  msnhNet;
		msnhNet.setUseFp16(true);			 //Inference with FP16
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

        int netX  = msnhNet.getInputSize().x;
        int netY  = msnhNet.getInputSize().y;

        std::vector<float> img = Msnhnet::CVUtil::getImgDataF32C3(imgPath,{netX,netY},false);
		std::vector<float> result;
		
		for (size_t i = 0; i < 10; i++)
		{
			auto st = Msnhnet::TimeUtil::startRecord();
			//msnhNet.runClassify(img); //Inference with CPU(not recommend)
			msnhNet.runClassifyGPU(img);
			std::cout << "time  : " << Msnhnet::TimeUtil::getElapsedTime(st) << "ms" << std::endl << std::flush;
		}
		
		Msnhnet::Mat mat(imgPath);

        Msnhnet::Mat mask(netX,netY,Msnhnet::MatType::MAT_RGB_U8);

        for (int i = 0; i < result.size()/2; ++i)
        {
            if(result[i] < result[i+msnhNet.getInputSize().x*msnhNet.getInputSize().y])
            {
                mask.getData().u8[i*3+0] += 120;
            }
        }

        Msnhnet::MatOp::resize(mask,mask,{mat.getWidth(),mat.getHeight()});
        mat = mat+mask;

        mat.saveImage("unet_res.jpg");
		std::cout << "\n**Result image will be save at root dir" << std::endl;
		getchar();
    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }

    return 0;
}
