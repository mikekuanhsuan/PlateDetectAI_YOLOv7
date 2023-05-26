import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication#傳送附件


def send_mail(recipient, subject, message, file = ''):
   
    username = "gx.kao@advanced-tek.com.tw"
    password = "Aa99405012!"
    
    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(message))
    
    file = file #附件路徑
    if len(file) > 0:
        part_attach1 = MIMEApplication(open(file,'rb').read())   #開啟附件
        part_attach1.add_header('Content-Disposition','attachment',filename=file) #為附件命名
        msg.attach(part_attach1)

    mailServer = smtplib.SMTP('smtp-mail.outlook.com', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(username, password)
    mailServer.sendmail(username, recipient, msg.as_string())
    mailServer.close()



if __name__ == '__main__':
    finish_txt = ("完成筆數")
    send_mail('gx.kao@advanced-tek.com.tw',"每15分鐘資料完成",finish_txt)
