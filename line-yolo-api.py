from __future__ import unicode_literals

import errno
import os
import sys
import tempfile
from dotenv import load_dotenv

from flask import Flask, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, PostbackEvent, StickerMessage, StickerSendMessage, 
    LocationMessage, LocationSendMessage, ImageMessage, ImageSendMessage)

import time
from pathlib import Path

import cv2
import torch
from utils.plots import Annotator, colors

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# reads the key-value pair from .env file and adds them to environment variable.
load_dotenv()

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')


### YOLOv5 ###
# Setup
weights, view_img, save_txt, imgsz = 'yolov5s.pt', False, False, 640
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
save_conf = False
save_img = True
line_thickness = 3

# Directories
save_dir = 'static/tmp/'

# Load model
model = torch.hub.load('./', 'custom', path='yolov5s.pt', source='local', force_reload=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# function for create tmp dir for download content
def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise

@app.route("/", methods=['GET'])
def home():
    return "Object Detection API"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text

    if text == 'profile':
        if isinstance(event.source, SourceUser):
            profile = line_bot_api.get_profile(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='Display name: ' + profile.display_name),
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't use profile API without user ID"))
    # else:
    #     line_bot_api.reply_message(
    #         event.reply_token, TextSendMessage(text=event.message.text))


@handler.add(MessageEvent, message=LocationMessage)
def handle_location_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        LocationSendMessage(
            title='Location', address=event.message.address,
            latitude=event.message.latitude, longitude=event.message.longitude
        )
    )


@handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        StickerSendMessage(
            package_id=event.message.package_id,
            sticker_id=event.message.sticker_id)
    )


# Other Message Type
@handler.add(MessageEvent, message=(ImageMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    else:
        return

    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.' + ext
    os.rename(tempfile_path, dist_path)

    im_file = open(dist_path, "rb")
    im = cv2.imread(im_file)
    im0 = im.copy()

    results = model(im, size=640)  # reduce size=320 for faster inference
    print(results)
    annotator = Annotator(im0, line_width=line_thickness)
    # Write results 
    df = results.pandas().xyxy[0]
    for idx, r in df.iterrows():
        c = int(r['class'])  # integer class
        name = r['name']
        label = f'{name} {r.confidence:.2f}'
        annotator.box_label((r.xmin, r.ymin, r.xmax, r.ymax), label, color=colors(c, True))
        

    save_path = str(save_dir + os.path.basename(tempfile_path) + '_result.' + ext) 
    cv2.imwrite(save_path, im0)

    url = request.url_root + '/' + save_path

    # line_bot_api.reply_message(
    #     event.reply_token, [
    #         TextSendMessage(text="ผลการตรวจโรค:"),
    #         ImageSendMessage(url,url),
    #     ])
    if name=="Anthracnose":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="ผลการตรวจโรค:"),
                ImageSendMessage(url,url),
                TextSendMessage(text=
'''
พบเจอโรค แอนเเทรคโนส (anthracnose)  เกิดจากเชื้อรา Colletotrichum Zibethinum ทำลายช่อดอกในระยะช่อบาน ทำให้ดอกมีสีคล้ำ เน่าดำก่อนบาน มีราสีเทาดำปกคลุมเกสร กลีบดอก ทำให้ดอกแห้ง ร่วงหล่น

วิธีรักษาโรค:
1.ตัดแต่งพุ่มให้โปร่ง
2.ฉีดพ่นด้วย mancozeb ผสมหรือสลับกับ carbendazim
3.ฉีดพ่นด้วย ผลิตภัณฑ์ ซุปเปอร์โวก้าโปรคีโตพลัส ในช่วงเย็น (ห้ามใช้สารเคมีกำจัดเชื้อรา ร่วมกับ คีโตพลัส)
4.เติมสารซุปเปอร์ซิลิคอนโวก้า 1 ช้อนโต๊ะ ในถังฉีด เพื่อเพิ่มประสิทธิภาพของ สารเคมี  

ยาที่ใช้ในการรักษาโรค:
1.จัมเปอร์
    https://shope.ee/6fErA9IocH
2.เค็นจิ
    https://shope.ee/hvk4PDUH
3.การ์แรต
    https://shope.ee/4pnCzc9Bnc
''')
            ])


    if name=="Leaf Spot":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="ผลการตรวจโรค:"),
                ImageSendMessage(url,url),
                TextSendMessage(text=
'''
พบเจอโรค ใบจุด (Leaf Spot) เกิดจากเชื้อราหลายชนิด โดย หากเป็นเชื้อ Colletotrichum sp. ซึ่งทำให้เกิดโรคแอนแทรคโนส ใบอ่อนจะมีสีซีดคล้ายโดนน้ำร้อนลวก ส่วนขยายพันธุ์เป็นจุดดำ ๆ ส่วนใบแก่เป็นจุดกลมขอบแผลสีเข้ม และมีการขยายขนาด

วิธีรักษาโรค:
1.ฉีดพ่นทุเรียนระยะใบอ่อน ด้วย สารกลุ่ม mancozeb ผสมกับกลุ่ม benzimidazole เช่น benomyl หรือ carbendazim
2.ฉีดพ่นด้วย ผลิตภัณฑ์ชีวภัณฑ์ ซุปเปอร์โวก้าโปรคีโตพลัส ในช่วงเย็น (ห้ามใช้สารเคมีกำจัดเชื้อรา ร่วมกับ คีโตพลัส)
3.เติมสารซุปเปอร์ซิลิคอนโวก้า 1 ช้อนโต๊ะ ในถังฉีด เพื่อเพิ่มประสิทธิภาพของสารเคมีและชึวภัณฑ์

ยาที่ใช้ในการรักษาโรค:
1.จัมเปอร์
    https://shope.ee/6fErA9IocH
2.เค็นจิ
    https://shope.ee/hvk4PDUH
3.การ์แรต
    https://shope.ee/4pnCzc9Bnc
''')
            ])
        
    if name=="Leaf Blight":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="ผลการตรวจโรค:"),
                ImageSendMessage(url,url),
                TextSendMessage(text=
'''
พบเจอโรค ใบติด,ใบไหม้,ใบร่วง (Leaf blight, leaf fall) เชื้อสัมผัสกับใบ ทำให้กิ่งเน่า เกิดจากเชื้อรา Rhizoctonia solani ลักษณะอาการใบจะไหม้ แห้ง และติดกันเป็นกระจุก และร่วงจำนวนมาก ใบติดกันด้วยเส้นใยของเชื้อรา ใบคล้ายถูกน้ำร้อนลวก สีซีด ขอบแผลสีเขียวเข้ม

สาเหตุของโรค:
-แดดเผา(Sun burn)
ใบยังไม่แก่พอ ทำให่ความต้านทานต่อแสงแดดมีน้อยเกิดอาการใบไหม้เมื่อมีอุณหภูมิสูว เป็นเวลานาน ใบจะมีอัตราการคายน้ำที่สูงจะเริ่มเหลืองและเกิดอาการใบไหม้
-การใส่ปุ๋ย สารเคมี
ผสมปุ๋ยยาที่เข้มข้นเกินไป และการใช้ยาร้อยในขณะยังเป็นใบอ่อนควรให้ปริมาณเหมาะสมกับสภาพของพืช
-เชื้อฟิวซาเรียม
ปลายใบทุเรียนจะเป็นใบแห้งๆ ส่องปลายใบกับแดดจะพบสปอร์เป็นขุยขาวๆ ของเชื้อรา ยอกทุเรียนที่แตกใหม่จะสักเกตุขุยสปอร์ทำให้ยอดแห้ง หากเป็นที่กึ่งจะลามรวดเร็วและทำให้กิ่งแห้ง
-การให้น้ำ
กระจายวงน้ำให้ทั่วอย่างน้อย80% ของทรงพุ่มโดยเฉพาะช่วงหน้าแล้งอากาศร้อนจัด ถ้าให้น้ำน้อยเกินไปก็จะไปส่งเสริมให้เกิดอาการใบไหม้"Sun Burn" ได้ง่ายขึ้น

วิธีรักษาโรค:
1.รวมรวมเศษใบที่ร่วงเผาทำลาย กำจัดวัชพืช
2.ฉีดพ่นด้วย copper oxychloride หรือ mancozeb
3.ฉีดพ่นด้วย ซุปเปอร์โวก้าโปรคีโตพลัส (ห้ามใช้สารเคมีกำจัดเชื้อรา ร่วมกับ คีโตพลัส)
4.เติมสารซุปเปอร์ซิลิคอนโวก้า 1 ช้อนโต๊ะ ในถังฉีด เพื่อเพิ่มประสิทธิภาพของ สารเคมี และชีวภัณฑ์
5.ควบคุมเชื้อราในดิน โดยการใช้ปุ๋ยโวก้าอินทรีย์ ประมาณ 5 กก. ผสมซุปเปอร์โวก้าโปรคีโตพลัส 50 กรัม คลุกให้ทั่ว นำปุ๋ยที่ได้หว่านรอบทรงพุ่มเพื่อควบคุมเชื้อรา

ยาที่ใช้ในการรักษาโรค
1.จัมเปอร์
    https://shope.ee/6fErA9IocH
2.เค็นจิ
    https://shope.ee/hvk4PDUH
3.การ์แรต
    https://shope.ee/4pnCzc9Bnc
''')
            ])
        
    if name=="Algal leaf Spot":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="ผลการตรวจโรค:"),
                ImageSendMessage(url,url),
                TextSendMessage(text=
'''
พบเจอโรค โรคราสนิม (Rust disease) เกิดจากสาหร่าย Cephaleuros virescens Kunze พบ ในใบแก่ ลักษณะเป็นจุดฟูเสีเขียวแกมเหลือง ต่อมาเปลี่ยนเป็นสีเหลืองแกมส้ม ซึ่งเป็นระยะที่สาหร่ายสร้างสปอร์ เพื่อใช้ในการแพร่ระบาด

วิธีรักษาโรค:
1.ฉีดพ่นด้วย Copper oxychloride
2.เติมสารซุปเปอร์ซิลิคอนโวก้า 1 ช้อนโต๊ะ ในถังฉีด เพื่อเพิ่มประสิทธิภาพของ สารเคมี และชีวภัณฑ์

ยาที่ใช้ในการรักษาโรค:
1.จอยท์
    https://shope.ee/fxe5gOXum
''')
            ])

    if name=="Healthy":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="ผลการตรวจโรค:"),
                ImageSendMessage(url,url),
                TextSendMessage(text=
                '''
                ใบของคุณมีสภาพแข็งแรงปกติ
                ''')
    
            ])

    else:
         line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text="ไม่ตรวจพบเจอโรค หรือ มีข้อผิดพลาด"),
            ])
         
    name=""
    

@app.route('/static/<path:path>')
def send_static_content(path):
    return send_from_directory('static', path)

# create tmp dir for download content
make_static_tmp_dir()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

