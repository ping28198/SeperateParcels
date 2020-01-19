#pragma once
//#include "pch.h"
#include <string>
#include <vector>
//#include "windows.h"

//////////////////////////////////////////////////////////////////////////
//该文件中只提供不依赖于第三方库的常用函数


class CommonFunc
{
public:
	CommonFunc();


	//************************************
	// 函数名:    getAllFilesNameInDir
	// 描述：    获取文件路径下的所有文件,可指定文件名后缀.不包括：“.”和“..”
	// 引用类型:    public static 
	// 返回值:   int :获得的文件名数量,没有获得文件名为0.
	// 参数: string dir :输入文件路径，可以是目录，也可以是带通配文件名，例如：/*.jpg /*.*
	// 参数: vector<string> & filenames :返回文件名信息
	// 参数: bool isIncludeSubDir :是否包括子目录，默认不包括
	// 参数: bool isReturnPath : 返回文件名中是否包括文件路径，默认不包括
	//************************************
	static int getAllFilesNameInDir(std::string dir, std::vector<std::string> &filenames,bool isIncludeSubDir=false, bool isReturnPath=false);
	static int getAllSubDirsInDir(std::string dir, std::vector<std::string> &subdirs, bool isIncludeSubDir = false, bool isReturnPath = false);
	
	//************************************
	// 函数:    get_exe_dir		
	// 全名:  CommonFunc::get_exe_dir		
	// 返回值:   std::string		#exe运行目录(d:\dir\)
	//************************************
	static std::string get_exe_dir();


	//************************************
	// 函数:    splitDirectoryAndFilename		
	// 作用： 将路径字符串分割成目录和文件名 #例如："D:/aa/bb.jpg" => "d:/aa/" +"bb.jpg"
	// 全名:  CommonFunc::splitDirectoryAndFilename		
	// 返回值:   int		
	// 参数: std::string src_full_path			#输入的全路径
	// 参数: std::string & dstDirectory			#输出的目录路径
	// 参数: std::string & dstFilename			#输出的文件名
	//************************************
	static int splitDirectoryAndFilename(std::string src_full_path, std::string &dstDirectory, std::string &dstFilename);

	//************************************
	// 函数:    joinFilePath		
	// 作用： 连接两个路径字符串，自动处理斜杠和反斜杠
	// 全名:  CommonFunc::joinFilePath		
	// 返回值:   int		#
	// 参数: std::string path1			#输入路径1
	// 参数: std::string path2			#输入路径1
	// 参数: std::string & dstFullPath			#输出全路径
	//************************************
	static int joinFilePath(std::string path1, std::string path2, std::string &dstFullPath);
	static std::string joinFilePath(std::string path1, std::string path2);

	//************************************
	// 函数:    getExtensionFilename		
	// 作用：获取文件的扩展名
	// 全名:  CommonFunc::getExtensionFilename		
	// 返回值:   int		#
	// 参数: std::string srcPath			#
	// 参数: std::string & dstExName			#
	//************************************
	static std::string getExtensionFilename(std::string srcPath);

	static std::string getShortFilename(std::string srcPath);

	//************************************
	// 函数:    WCharToChar		
	// 作用：将宽字节字符转为多字节字符,支持最大字符长度4096,返回需要使用string（CString）接收。
	// 全名:  CommonFunc::WCharToChar		
	// 返回值:   const char*		#
	// 参数: const wchar_t * srcWChar			#
	//************************************
	static const char* WCharToMChar(const wchar_t* srcWChar);
	static const wchar_t* MCharToWChar(const char* srcChar);
};