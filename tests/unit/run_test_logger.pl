#!/usr/bin/env perl
use warnings;
use strict;
use v5.16;
use Text::ParseWords qw(shellwords);

die "USAGE: $0 <path to mpirun> <maximum nprocs> <path to test_logger executable>" unless (3 == scalar @ARGV);

my ($mpicmd, $maxprocs, $target) = @ARGV;
my @mpicmd = shellwords($mpicmd);

my @fails = ();
foreach my $nprocs (1..$maxprocs) {
  open(my $pipe, "-|", @mpicmd, "-n", $nprocs, $target)
    or die "Cannot run logger test: $!";
  my $out = do { local $/; <$pipe> };
  close($pipe);

  # Keep the exit code of the subprocess
  my $ec = $?;

  # Trim newlines from the output
  chomp($out);

  # See test_logger.cpp for these values
  my $xs = scalar grep /XXX/, (split '\n', $out);
  my $ys = scalar grep /YYY/, (split '\n', $out);

  if (1 != $xs or $nprocs != $ys or $ec) {
    push @fails, $nprocs;
  }
}

if (scalar @fails) {
  say "FAIL: ExaGOLog failed when tested with the following number of processes:";
  say join "\n", @fails;
  exit 1;
}

exit 0;
