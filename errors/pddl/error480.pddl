(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(clear b4)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b8 t0)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b5 b6)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(yellow b0)
		(green b1)
		(yellow b3)
		(green b4)
		(blue b4)
		(yellow b5)
		(green b6)
		(blue b6)
		(yellow b7)
		(green b8)
		(green b9)
		(yellow b1)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (green ?x)) (exists (?y) (and (yellow ?y) (on ?x ?y))))) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b9 t0)) (not (on b9 b8)) (not (on b9 b7)) (not (on b9 b6)) (not (on b9 b5)) (not (on b9 b4)))))
)