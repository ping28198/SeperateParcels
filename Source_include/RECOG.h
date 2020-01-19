/* 
版本号：  1.2.0.3
发行日期：2013-07-11  
作者：    徐海堰 
*/
   
#if !defined(RECOG_h)
#define RECOG_h

// 打印体识别类
class C_RECOG
{
private:
	BYTE *pBIMG;  // 输入的二值化图像指针（从左往右从上往下每个像素点1个字节）
	WORD wPixels_X,wPixels_Y;  

public:
/////////////////////////////////////////////////////////////////////////////
// 打印体识别相关成员变量 

	BYTE bFlag_HWPrint;         // 0xff-default; 0-TRANS OK; 1,2-TRANS error;
	                            // 3-Print RECOG OK; 4-HW_SlantImage() error; 5-LineAndWordSegment() error.
    BYTE *pBIMG_HWPrint;        // 供识别的二值化图像指针（从左往右从上往下每8个像素点1个字节）
	RECT RECT_HWPrint;    
	WORD wPixels_X_HWPrint,wPixels_Y_HWPrint;  // 供识别的二值化图像XY方向像素数量
	DWORD dwTime_HWPrint;       // 识别耗时（单位：毫秒）
	CString sFullPath_HWPrint;  // 工作目录（其中应包含必要的支撑文件）
	
	// 识别结果相关变量
	BYTE bRESRowNUM_HWPrint;    // 识别结果文本行的数量（不包含空文本行）
	CString asRES_HWPrint[100];
	CString sRES_HWPrint;

/////////////////////////////////////////////////////////////////////////////

public:
	C_RECOG(BYTE *mpBIMG,WORD mwPixels_X,WORD mwPixels_Y);
	~C_RECOG();
	void Reset_Check();

	void TRANS_HWPrint_Check(RECT mRECT);
	void RECOG_HWPrint_Check(char *mPath);
	CString GetRECOGRES();
};

#endif

/*
示例：

class C_RECOG RECOG(pBinaryImage,wLength,wWidth);
RECT R;

// 识别范围设为整幅图像
R.left=R.top=0;
R.right=wLength-1;
R.bottom=wWidth-1; 

RECOG.TRANS_HWPrint_Check(R);  
RECOG.RECOG_HWPrint_Check("c:\\OCR\\");
if (RECOG.bFlag_HWPrint==3) RECOG.GetRECOGRES();
RECOG.Reset_Check();  // 所有处理结束后重置
*/