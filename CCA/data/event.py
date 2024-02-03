import pandas as pd


class Event:
    """
    Class for a collision event from the TrackML dataset
    """

    default_path_to_base = "/Users/archibaldruban/Google Drive/1. Education/UCL/1. BSc Theoretical Physics/_year 4/Master's Project/datasets/train_100_events"

    def __init__(self, event_id, path_to_base=None):
        self.event_id = event_id
        self.data = {"cells": None, "hits": None, "particles": None, "truth": None}

        self.path_to_base = (
            path_to_base if path_to_base != None else self.default_path_to_base
        )

    def load_content(self, content_to_load):
        """
        Loads the content of .csv file into self.data

        content_to_load options: "all", cells", "hits", "particles", "truth"
        """
        if content_to_load == "all":
            for key in self.data.keys():
                path = self.get_path_to(key)
                self.data[key] = pd.read_csv(path)

        else:
            path = self.get_path_to(content_to_load)
            self.data[key] = pd.read_csv(path)

    def get_path_to(self, content_name):
        """
        Gives the path to the .csv file that contains content 
        specified by fileContent

        content_name options: "cells", "hits", "particles", "truth", "base"
        """
        if content_name == "base":
            return self.path_to_base

        return self.path_to_base + "/" + self.event_id + "-" + content_name + ".csv"
