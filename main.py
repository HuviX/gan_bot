import argparse
import logging
import time
import datetime
from net import Generator
import sys
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='weights/', help='model weights')
opt = parser.parse_args()



class myBot:
    def __init__(self, token):
        self._init_bot(token)
        self._init_model()
        self._init_handlers()
        print("Bot is ready")


    def _init_bot(self, token):
        self._token = token
        self._bot = telegram.Bot(token=self._token)
        self._updater = Updater(token=self._token)
        self._dispatcher = self._updater.dispatcher

        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    def _init_model(self):
        self.model_woman = Generator()
        self.model_woman.load(opt.path + 'female.pth')
        self.model_man = Generator()
        self.model_man.load(opt.path + "male.pth")


    def _init_handlers(self):
        def echo(bot, update):
            self._bot.send_message(chat_id=update.message.chat_id, text="Со мной нельзя общаться :(")


        def start(bot, update):
            self._bot.send_sticker(chat_id=update.message.chat_id, sticker='CAACAgIAAxkBAAEBZqlfdkxReii4gnPpc-89Af3KjPhsHwACggADtSkaGb_7zPJFap3GGwQ')
            self._bot.send_message(chat_id=update.message.chat_id, text ="Команда /woman возвращает случайную женщину из GANa, /man - мужчину")

        def time(bot, update):
            text = str(datetime.datetime.utcnow())
            self._bot.send_message(chat_id=update.message.chat_id, text=text)


        def gen_woman(bot, update):
            self.model_woman.predict()
            self._bot.send_photo(chat_id=update.message.chat_id, photo=open('fromBot.png','rb'))

        def gen_man(bot, update):
            self.model_man.predict()
            self._bot.send_photo(chat_id=update.message.chat_id, photo=open('fromBot.png','rb'))


        self._handlers = [CommandHandler(['hi','privet','help'], start),
                          CommandHandler(['time'], time),
                          CommandHandler(['woman'],gen_woman),
                          CommandHandler(['man'], gen_man),
                          MessageHandler(Filters.text, echo)]
        for handler in self._handlers:
            self._dispatcher.add_handler(handler)
    
    def start(self):
        self._updater.start_polling()


    def stop(self):
        self._updater.stop()


if __name__ == '__main__':
    with open("token.txt") as f:
        token = f.read().strip()
    bot = myBot(token)
    bot.start()
