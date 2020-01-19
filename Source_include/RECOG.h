/* 
�汾�ţ�  1.2.0.3
�������ڣ�2013-07-11  
���ߣ�    �캣�� 
*/
   
#if !defined(RECOG_h)
#define RECOG_h

// ��ӡ��ʶ����
class C_RECOG
{
private:
	BYTE *pBIMG;  // ����Ķ�ֵ��ͼ��ָ�루�������Ҵ�������ÿ�����ص�1���ֽڣ�
	WORD wPixels_X,wPixels_Y;  

public:
/////////////////////////////////////////////////////////////////////////////
// ��ӡ��ʶ����س�Ա���� 

	BYTE bFlag_HWPrint;         // 0xff-default; 0-TRANS OK; 1,2-TRANS error;
	                            // 3-Print RECOG OK; 4-HW_SlantImage() error; 5-LineAndWordSegment() error.
    BYTE *pBIMG_HWPrint;        // ��ʶ��Ķ�ֵ��ͼ��ָ�루�������Ҵ�������ÿ8�����ص�1���ֽڣ�
	RECT RECT_HWPrint;    
	WORD wPixels_X_HWPrint,wPixels_Y_HWPrint;  // ��ʶ��Ķ�ֵ��ͼ��XY������������
	DWORD dwTime_HWPrint;       // ʶ���ʱ����λ�����룩
	CString sFullPath_HWPrint;  // ����Ŀ¼������Ӧ������Ҫ��֧���ļ���
	
	// ʶ������ر���
	BYTE bRESRowNUM_HWPrint;    // ʶ�����ı��е����������������ı��У�
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
ʾ����

class C_RECOG RECOG(pBinaryImage,wLength,wWidth);
RECT R;

// ʶ��Χ��Ϊ����ͼ��
R.left=R.top=0;
R.right=wLength-1;
R.bottom=wWidth-1; 

RECOG.TRANS_HWPrint_Check(R);  
RECOG.RECOG_HWPrint_Check("c:\\OCR\\");
if (RECOG.bFlag_HWPrint==3) RECOG.GetRECOGRES();
RECOG.Reset_Check();  // ���д������������
*/