import requests
from bs4 import BeautifulSoup
import re
from urllib import request
from time import sleep

save_path = r"E:\Files\wallpaper"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleW"
                  "ebKit/537.36 (KHTML,like Gecko) Chrome/70.0.3538.110 Safari/537.36"}
domain_url = "http://pic.netbian.com/"


def get_2(url):
    res = requests.get(url, headers=headers).content
    bs_obj = BeautifulSoup(res, "html5lib")
    return bs_obj


def get_next():
    for i in range(0, 15):
        bs_obj = get_2("http://pic.netbian.com/e/search/result/?searchid=2252&page=%s" % i)
        my_re = '<li><a href="(.+?)" target="_blank">'
        find_res = re.findall(re.compile(my_re), str(bs_obj))
        for each in find_res:
            print(each[8:each.find(".")])
            imgurl = domain_url + each
            pic_page = get_2(imgurl)

            my_re_2 = 'src="(.+?)"'
            try:
                find_res_2 = pic_page.find("a", id="img")
                print(find_res_2)
                pic_url = re.search(re.compile(my_re_2), str(find_res_2)).group(1)
                imageurl = domain_url + pic_url[1:]
                if imageurl:
                    print(imageurl)
                    with open(r'E:\Files\wallpaper\%s.jpg' % each[8:each.find(".")], "wb+") as f:
                        f.write(requests.get(imageurl).content)
                    print("done!!")
            except BaseException:
                print("Error")
            sleep(2)
        sleep(5)


if __name__ == '__main__':
    get_next()
