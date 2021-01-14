from corpus import corpus
import time


def main() -> None:
    '''
    Main function for bot.
    '''

    c = corpus()
    c.get_all_text_data(all_in_one=True)


if __name__ == '__main__':
    start_time: float = time.time()
    main()
    duration: float = time.time() - start_time
    print(f'Done: {duration:.2f} s')
