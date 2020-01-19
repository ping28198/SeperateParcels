#include "CommonFunc.h"
#include "tchar.h"
#include <iostream>
#include <atlconv.h>

using namespace std;

CommonFunc::CommonFunc()
{

}


int CommonFunc::getAllFilesNameInDir(string dir, vector<string> &filenames, bool isIncludeSubDir/*=false*/, bool isReturnPath/*=false*/)
{
	HANDLE hFind;
	WIN32_FIND_DATA findData;
	//LARGE_INTEGER size;
	string base_dir,new_dir,suffix;
	//dir.replace(dir.begin(), dir.end(), '\\', '/');
	while (true)
	{
		int mpos = dir.find('\\');
		if (mpos!=dir.npos)
		{
			dir.replace(mpos, 1, "/");
		}
		else
		{
			break;
		}
	}
	while (true)
	{
		if (dir.find_last_of('/')!=(dir.size()-1))
		{
			break;
		}
		dir.erase(dir.size() - 1);
	}
	base_dir = dir;
	if (base_dir.find('*') != base_dir.npos || base_dir.find('?') != base_dir.npos)
	{
		int slashPos = dir.find_last_of('/');
		if (slashPos!=dir.npos)
		{
			suffix = base_dir.substr(slashPos+1);
			base_dir = base_dir.substr(0, slashPos);
		}
	}
	else
	{
		suffix = "*.*";
	}
	if (isIncludeSubDir)
	{
		new_dir = base_dir + "/" + "*.*";
	}
	else
	{
		new_dir = base_dir + "/" + suffix;
	}
	USES_CONVERSION;
	hFind = FindFirstFileW(A2T(new_dir.c_str()), &findData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		//cout << "Failed to find first file!\n";
	}
	else
	{
		string str1;
		char tmp[256] = { 0 };
		do
		{
			// 忽略"."和".."两个结果 
			WideCharToMultiByte(CP_ACP, 0, findData.cFileName, 256, tmp, 256, 0, 0);
			str1 = std::string(tmp);
			if (strcmp(str1.c_str(), ".") == 0 || strcmp(str1.c_str(), "..") == 0)
				continue;
			if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)    // 是否是目录 
			{
				if (isIncludeSubDir)
				{
					getAllFilesNameInDir(base_dir + '/' + str1+'/'+suffix, filenames, isIncludeSubDir, isReturnPath);
				}
			}
		} while (FindNextFile(hFind, &findData));
	}

	//查找文件
	new_dir.clear();
	new_dir = base_dir + "/" + suffix;
	hFind = FindFirstFileW(A2T(new_dir.c_str()), &findData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		cout << "Failed to find first file!\n";
	}
	else
	{
		std::string str1;
		char tmp[256] = { 0 };
		do
		{
			//WideCharToMultiByte(CP_ACP, 0, findData.cFileName, 256, tmp, 256, 0, 0);
			//str1 = std::string(tmp);
			str1 = CommonFunc::WCharToMChar(findData.cFileName);
			if (strcmp(str1.c_str(), ".") == 0 || strcmp(str1.c_str(), "..") == 0)
				continue;
			if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)    // 是否是目录 
			{
				continue;
			}
			else
			{
				if (isReturnPath)
				{
					filenames.push_back(base_dir + '/' + str1);
				}
				else
				{
					filenames.push_back(str1);
				}
			}
		} while (FindNextFile(hFind, &findData));
	}

	return filenames.size();
}

int CommonFunc::getAllSubDirsInDir(std::string dir, std::vector<std::string> &subdirs, bool isIncludeSubDir /*= false*/, bool isReturnPath /*= false*/)
{
	HANDLE hFind;
	WIN32_FIND_DATA findData;
	//LARGE_INTEGER size;
	string base_dir, new_dir, suffix;
	//dir.replace(dir.begin(), dir.end(), '\\', '/');
	while (true)
	{
		int mpos = dir.find('\\');
		if (mpos != dir.npos)
		{
			dir.replace(mpos, 1, "/");
		}
		else
		{
			break;
		}
	}
	while (true)
	{
		if (dir.find_last_of('/') != (dir.size() - 1))
		{
			break;
		}
		dir.erase(dir.size() - 1);
	}
	base_dir = dir;
	if (base_dir.find('*') != base_dir.npos || base_dir.find('?') != base_dir.npos)
	{
		int slashPos = dir.find_last_of('/');
		if (slashPos != dir.npos)
		{
			suffix = base_dir.substr(slashPos + 1);
			base_dir = base_dir.substr(0, slashPos);
		}
	}
	else
	{
		suffix = "*.*";
	}
	if (isIncludeSubDir)
	{
		new_dir = base_dir + "/" + "*.*";
	}
	else
	{
		new_dir = base_dir + "/" + suffix;
	}
	USES_CONVERSION;
	hFind = FindFirstFileW(A2T(new_dir.c_str()), &findData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		//cout << "Failed to find first file!\n";
	}
	else
	{
		do
		{
			// 忽略"."和".."两个结果
			string str1 = T2A(findData.cFileName);
			if (strcmp(str1.c_str(), ".") == 0 || strcmp(str1.c_str(), "..") == 0)
				continue;
			if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)    // 是否是目录 
			{
				if (isReturnPath)
				{
					subdirs.push_back(base_dir + '/' + str1);
				}
				else
				{
					subdirs.push_back(str1);
				}
				//if (isIncludeSubDir)
				//{
				//	getAllFilesNameInDir(base_dir + '/' + str1 + '/' + suffix, filenames, isIncludeSubDir, isReturnPath);
				//}
			}
		} while (FindNextFile(hFind, &findData));
	}


	return subdirs.size();
}

std::string CommonFunc::get_exe_dir()
{
	TCHAR szFilePath[MAX_PATH + 1] = { 0 };
	GetModuleFileName(NULL, szFilePath, MAX_PATH);
	(_tcsrchr(szFilePath, _T('\\')))[1] = 0; // 删除文件名，只获得路径字串
	//str_url = szFilePath;  // 例如str_url==e:\program\Debug\  //
	USES_CONVERSION;
	std::string exe_dir = T2A(szFilePath);
	return exe_dir;
}

int CommonFunc::splitDirectoryAndFilename(std::string src_full_path, std::string &dstDirectory, std::string &dstFilename)
{

	size_t pos1 = src_full_path.find_last_of('\\');
	size_t pos2 = src_full_path.find_last_of('/');
	size_t pos;
	if (pos1!=src_full_path.npos && pos2 != src_full_path.npos)
	{
		pos = (pos1 < pos2) ? pos2 : pos1;
		pos++;
	}
	else if(pos1!= src_full_path.npos)
	{
		pos = pos1;
		pos++;
	}
	else if (pos2 != src_full_path.npos)
	{
		pos = pos2;
		pos++;
	}
	else
	{
		pos = 0;
	}
	dstDirectory = src_full_path.substr(0, pos);
	dstFilename = src_full_path.substr(pos);
	return 1;
}

int CommonFunc::joinFilePath(std::string path1, std::string path2, std::string &dstFullPath)
{
	size_t pos1 = path1.find_last_of('\\');
	size_t pos2 = path1.find_last_of('/');
	size_t pos3 = path2.find_first_of('\\');
	size_t pos4 = path2.find_first_of('/');
	if ((pos1!=(path1.size()-1)) && (pos2 != (path1.size()-1)) && (pos3 != 0) && (pos4 != 0))
	{
		path1.append("/");
	}
	else if(((pos1 == (path1.size() - 1)) || (pos2 == (path1.size() - 1)))
		&& ((pos3 == 0) || (pos4 == 0)))
	{
		path1.pop_back();
	}
	dstFullPath = path1 + path2;
	return 1;
}

std::string CommonFunc::joinFilePath(std::string path1, std::string path2)
{
	string joinedpath;
	joinFilePath(path1, path2, joinedpath);
	return joinedpath;
}

std::string CommonFunc::getExtensionFilename(std::string srcPath)
{
	size_t pos1 = srcPath.find_last_of('\\');
	size_t pos2 = srcPath.find_last_of('/');
	size_t pos;
	if (pos1 != srcPath.npos && pos2 != srcPath.npos)
	{
		pos = (pos1 < pos2) ? pos2 : pos1;
		pos++;
	}
	else if (pos1 != srcPath.npos)
	{
		pos = pos1;
		pos++;
	}
	else if (pos2 != srcPath.npos)
	{
		pos = pos2;
		pos++;
	}
	else
	{
		pos = 0;
	}
	string _filename = srcPath.substr(pos);
	string ExFilename = "";
	pos = _filename.find_last_of('.');
	if (pos != _filename.npos)
	{
		ExFilename = _filename.substr(pos + 1);
	}
	return ExFilename;
}

std::string CommonFunc::getShortFilename(std::string srcPath)
{
	std::string filename;
	std::string s;
	splitDirectoryAndFilename(srcPath, s, filename);
	size_t pos = filename.find_last_of('.');
	string shortname;
	if (pos != filename.npos)
	{
		shortname = filename.substr(0, pos);
	}
	else
	{
		shortname = filename;
	}
	return shortname;
}

const char* CommonFunc::WCharToMChar(const wchar_t* srcWChar)
{
	size_t wlength = wcslen(srcWChar);
	char tmp[4096] = { 0 };
	WideCharToMultiByte(CP_ACP, 0, srcWChar, wlength, tmp, 4096, NULL, NULL);
	return tmp;
}

const wchar_t* CommonFunc::MCharToWChar(const char* srcChar)
{
	size_t slen = strlen(srcChar);
	wchar_t wtmp[4096] = { 0 };
	MultiByteToWideChar(CP_ACP, 0, srcChar, slen, wtmp, 4096);
	return wtmp;
}

