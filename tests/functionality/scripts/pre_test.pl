#!/usr/bin/env perl
use strict;
use warnings;
use File::Glob qw(bsd_glob);

unlink bsd_glob("$ARGV[0]/*.warning") if @ARGV;
