from corpus import corpus


def main() -> None:
    '''
    Main function for bot.
    '''

    c = corpus()
    c.get_all_text_data(all_in_one=True)
    c.read_data()


if __name__ == '__main__':
    main()
