#pragma once
//#include "pch.h"
#include <string>
#include <vector>
//#include "windows.h"

//////////////////////////////////////////////////////////////////////////
//���ļ���ֻ�ṩ�������ڵ�������ĳ��ú���


class CommonFunc
{
public:
	CommonFunc();


	//************************************
	// ������:    getAllFilesNameInDir
	// ������    ��ȡ�ļ�·���µ������ļ�,��ָ���ļ�����׺.����������.���͡�..��
	// ��������:    public static 
	// ����ֵ:   int :��õ��ļ�������,û�л���ļ���Ϊ0.
	// ����: string dir :�����ļ�·����������Ŀ¼��Ҳ�����Ǵ�ͨ���ļ��������磺/*.jpg /*.*
	// ����: vector<string> & filenames :�����ļ�����Ϣ
	// ����: bool isIncludeSubDir :�Ƿ������Ŀ¼��Ĭ�ϲ�����
	// ����: bool isReturnPath : �����ļ������Ƿ�����ļ�·����Ĭ�ϲ�����
	//************************************
	static int getAllFilesNameInDir(std::string dir, std::vector<std::string> &filenames,bool isIncludeSubDir=false, bool isReturnPath=false);
	static int getAllSubDirsInDir(std::string dir, std::vector<std::string> &subdirs, bool isIncludeSubDir = false, bool isReturnPath = false);
	
	//************************************
	// ����:    get_exe_dir		
	// ȫ��:  CommonFunc::get_exe_dir		
	// ����ֵ:   std::string		#exe����Ŀ¼(d:\dir\)
	//************************************
	static std::string get_exe_dir();


	//************************************
	// ����:    splitDirectoryAndFilename		
	// ���ã� ��·���ַ����ָ��Ŀ¼���ļ��� #���磺"D:/aa/bb.jpg" => "d:/aa/" +"bb.jpg"
	// ȫ��:  CommonFunc::splitDirectoryAndFilename		
	// ����ֵ:   int		
	// ����: std::string src_full_path			#�����ȫ·��
	// ����: std::string & dstDirectory			#�����Ŀ¼·��
	// ����: std::string & dstFilename			#������ļ���
	//************************************
	static int splitDirectoryAndFilename(std::string src_full_path, std::string &dstDirectory, std::string &dstFilename);

	//************************************
	// ����:    joinFilePath		
	// ���ã� ��������·���ַ������Զ�����б�ܺͷ�б��
	// ȫ��:  CommonFunc::joinFilePath		
	// ����ֵ:   int		#
	// ����: std::string path1			#����·��1
	// ����: std::string path2			#����·��1
	// ����: std::string & dstFullPath			#���ȫ·��
	//************************************
	static int joinFilePath(std::string path1, std::string path2, std::string &dstFullPath);
	static std::string joinFilePath(std::string path1, std::string path2);

	//************************************
	// ����:    getExtensionFilename		
	// ���ã���ȡ�ļ�����չ��
	// ȫ��:  CommonFunc::getExtensionFilename		
	// ����ֵ:   int		#
	// ����: std::string srcPath			#
	// ����: std::string & dstExName			#
	//************************************
	static std::string getExtensionFilename(std::string srcPath);

	static std::string getShortFilename(std::string srcPath);

	//************************************
	// ����:    WCharToChar		
	// ���ã������ֽ��ַ�תΪ���ֽ��ַ�,֧������ַ�����4096,������Ҫʹ��string��CString�����ա�
	// ȫ��:  CommonFunc::WCharToChar		
	// ����ֵ:   const char*		#
	// ����: const wchar_t * srcWChar			#
	//************************************
	static const char* WCharToMChar(const wchar_t* srcWChar);
	static const wchar_t* MCharToWChar(const char* srcChar);
};